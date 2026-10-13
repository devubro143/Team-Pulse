"""Automated tests for BolKeBatao AI audio buffering, silence detection, and pipeline.
Tests quiet speech, normal speech, initial silence, pauses between words, long continuous speech (>=10s),
sample-rate resampling (48kHz -> 16kHz mono), audio container decoding, and transcription quality guarding.
"""
import io
import time
import wave
import unittest
import numpy as np
import av

from app import (
    SilenceDetectingRecorder,
    frames_to_wav,
    normalize_audio_to_16k_mono_wav,
    inspect_audio_wav,
    evaluate_transcription_quality,
    validate_evidence,
    SILENCE_RMS_THRESHOLD,
    SILENCE_SECONDS,
    MIN_RECORDING_SECONDS,
    MAX_RECORDING_SECONDS,
)


def create_audio_frame(samples_np, sample_rate=48000, is_planar_stereo=False):
    """Helper to create an av.AudioFrame from float numpy samples."""
    if is_planar_stereo:
        stereo = np.stack([samples_np, samples_np])
        frame = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
    else:
        mono = samples_np.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(mono, format="flt", layout="mono")
    frame.sample_rate = sample_rate
    return frame


class TestAudioRecordingAndPipeline(unittest.TestCase):

    def test_initial_silence_does_not_stop_recording(self):
        """Initial microphone silence must not cause premature stop."""
        recorder = SilenceDetectingRecorder()
        rate = 48000
        chunk_size = 960  # 20ms
        silent_chunk = np.zeros(chunk_size, dtype=np.float32)

        # Simulate 2.5 seconds of initial silence (125 frames)
        for _ in range(125):
            f = create_audio_frame(silent_chunk, sample_rate=rate, is_planar_stereo=True)
            recorder.recv(f)

        frames, finished, err, last_rms, peak_rms, has_speech, count = recorder.get_recording()
        self.assertFalse(finished, "Recording should NOT finish during initial silence!")
        self.assertFalse(has_speech, "Speech should not be detected for zeros.")
        self.assertEqual(count, 125, "All 125 frames must be buffered.")
        self.assertEqual(err, None)

    def test_quiet_frame_or_rerun_does_not_interrupt_active_speech(self):
        """Single quiet frame or low volume during speech must not stop recording."""
        recorder = SilenceDetectingRecorder()
        rate = 48000
        chunk_size = 960

        # 1. User starts speaking (RMS ~ 0.05)
        t = np.linspace(0, 0.02, chunk_size, endpoint=False)
        speech_chunk = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
        f_speech = create_audio_frame(speech_chunk, sample_rate=rate, is_planar_stereo=True)
        recorder.recv(f_speech)

        frames, finished, err, last_rms, peak_rms, has_speech, count = recorder.get_recording()
        self.assertTrue(has_speech, "Speech must be detected.")
        self.assertFalse(finished)

        # 2. Brief pause or quiet frame (1 frame of silence = 20ms)
        f_quiet = create_audio_frame(np.zeros(chunk_size, dtype=np.float32), sample_rate=rate)
        recorder.recv(f_quiet)

        frames, finished, err, last_rms, peak_rms, has_speech, count = recorder.get_recording()
        self.assertFalse(finished, "A single quiet frame must not stop recording!")
        self.assertEqual(count, 2)

    def test_pauses_between_words_do_not_stop_recording_under_threshold(self):
        """Natural pause between words (e.g. 1.0 second) should not stop recording."""
        recorder = SilenceDetectingRecorder()
        rate = 48000
        chunk_size = 960
        t = np.linspace(0, 0.02, chunk_size, endpoint=False)
        speech_chunk = (np.sin(2 * np.pi * 440 * t) * 0.08).astype(np.float32)
        silent_chunk = np.zeros(chunk_size, dtype=np.float32)

        # Speak for 0.5s (25 frames)
        for _ in range(25):
            recorder.recv(create_audio_frame(speech_chunk, rate, is_planar_stereo=True))

        # Pause for 1.0s (50 frames of silence)
        for _ in range(50):
            recorder.recv(create_audio_frame(silent_chunk, rate, is_planar_stereo=True))

        frames, finished, err, last_rms, peak_rms, has_speech, count = recorder.get_recording()
        self.assertTrue(has_speech)
        self.assertFalse(finished, "1s pause between words should NOT stop recording (threshold is 3s).")
        self.assertEqual(count, 75)

    def test_continuous_speech_lasting_10_seconds(self):
        """Continuous speech lasting at least 10 seconds must all be buffered cleanly."""
        recorder = SilenceDetectingRecorder()
        rate = 48000
        chunk_size = 960
        t = np.linspace(0, 0.02, chunk_size, endpoint=False)
        speech_chunk = (np.sin(2 * np.pi * 300 * t) * 0.05).astype(np.float32)

        # 10 seconds = 500 frames of 960 samples
        for _ in range(500):
            recorder.recv(create_audio_frame(speech_chunk, rate, is_planar_stereo=True))

        frames, finished, err, last_rms, peak_rms, has_speech, count = recorder.get_recording()
        self.assertFalse(finished, "10 seconds of active speech must NOT be auto-stopped.")
        self.assertTrue(has_speech)
        self.assertEqual(count, 500, "All 500 frames must be retained in order.")

        # Finalize and convert to WAV (target rate 16000)
        recorder.mark_finished()
        wav_bytes = frames_to_wav(frames, target_rate=16000, trim_excess_silence=False)
        inspection = inspect_audio_wav(wav_bytes)

        self.assertTrue(inspection["is_valid"])
        self.assertAlmostEqual(inspection["duration"], 10.0, delta=0.1)
        self.assertEqual(inspection["channels"], 1)
        self.assertEqual(inspection["rate"], 16000)
        self.assertEqual(inspection["bit_depth"], 16)

    def test_sustained_silence_after_speech_auto_stops(self):
        """3.0+ seconds of silence AFTER speech should cleanly finish recording."""
        recorder = SilenceDetectingRecorder()
        rate = 48000
        chunk_size = 960
        t = np.linspace(0, 0.02, chunk_size, endpoint=False)
        speech_chunk = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
        silent_chunk = np.zeros(chunk_size, dtype=np.float32)

        # Speak for 0.5s (25 frames)
        for _ in range(25):
            recorder.recv(create_audio_frame(speech_chunk, rate, is_planar_stereo=True))

        # Manually verify speech was detected
        self.assertTrue(recorder.has_detected_speech)

        # Now simulate 3.2 seconds of silence
        recorder.started_at = time.monotonic() - 4.0
        recorder.last_voice_at = time.monotonic() - 3.1  # 3.1s ago

        # Feed 1 more silent frame
        recorder.recv(create_audio_frame(silent_chunk, rate, is_planar_stereo=True))

        frames, finished, err, last_rms, peak_rms, has_speech, count = recorder.get_recording()
        self.assertTrue(finished, "Should finish when quiet_for >= SILENCE_SECONDS.")

    def test_audio_validation_guards(self):
        """Audio validator rejects pure silence or too-short recordings before transcription."""
        rate = 16000

        # 1. Too short (0.2s)
        bio_short = io.BytesIO()
        with wave.open(bio_short, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(np.zeros(int(rate * 0.2), dtype=np.int16).tobytes())
        val_short = inspect_audio_wav(bio_short.getvalue())
        self.assertFalse(val_short["is_valid"])
        self.assertIn("too short", val_short["error"])

        # 2. Pure silence (2.0s)
        bio_silence = io.BytesIO()
        with wave.open(bio_silence, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(np.zeros(int(rate * 2.0), dtype=np.int16).tobytes())
        val_silence = inspect_audio_wav(bio_silence.getvalue())
        self.assertFalse(val_silence["is_valid"])
        self.assertIn("No speech was detected", val_silence["error"])

        # 3. Valid normal speech audio (2.0s)
        t = np.linspace(0, 2.0, int(rate * 2.0), endpoint=False)
        tone = (np.sin(2 * np.pi * 300 * t) * 3000).astype(np.int16)
        bio_valid = io.BytesIO()
        with wave.open(bio_valid, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(tone.tobytes())
        val_valid = inspect_audio_wav(bio_valid.getvalue())
        self.assertTrue(val_valid["is_valid"])
        self.assertAlmostEqual(val_valid["duration"], 2.0, delta=0.05)

    def test_evidence_validation_logic(self):
        """Verify clinical extraction anti-hallucination validation."""
        transcript = "Mera sir dard do din se hai aur paracetamol liya tha."
        mock_data = {
            "fields": {
                "complaint": {"value": "sir dard", "evidence": "sir dard", "status": "supported"},
                "duration": {"value": "2 days", "evidence": "do din", "status": "supported"},
                "medicines": {"value": "paracetamol", "evidence": "paracetamol", "status": "supported"},
                "allergies": {"value": None, "evidence": None, "status": "not_mentioned"},
                "severity": {"value": "severe", "evidence": "bahut tez", "status": "supported"},  # not in transcript
            }
        }
        res = validate_evidence(mock_data, transcript)
        self.assertIn("Verified", res["fields"]["complaint"]["validation"])
        self.assertIn("Verified", res["fields"]["duration"]["validation"])
        self.assertIn("Not mentioned", res["fields"]["allergies"]["validation"])
        # 'bahut tez' is not in transcript, should be flagged
        self.assertIn("Warning", res["fields"]["severity"]["validation"])

    def test_wav_resampling_integrity_and_sample_rate_conversion(self):
        """Verify that PyAV AudioFrames at 48kHz stereo are correctly resampled to 16kHz mono 16-bit PCM."""
        rate = 48000
        target_rate = 16000
        chunk_size = 960  # 20ms
        t = np.linspace(0, 0.02, chunk_size, endpoint=False)
        tone_chunk = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        # 50 frames = 1.0 second
        frames = [
            create_audio_frame(tone_chunk, sample_rate=rate, is_planar_stereo=True)
            for _ in range(50)
        ]
        wav_bytes = frames_to_wav(frames, target_rate=target_rate, trim_excess_silence=False)
        insp = inspect_audio_wav(wav_bytes)

        self.assertTrue(insp["is_valid"], f"Inspection failed: {insp.get('error')}")
        self.assertEqual(insp["rate"], 16000, "WAV framerate must be exactly 16000 Hz.")
        self.assertEqual(insp["channels"], 1, "WAV must be mono (1 channel).")
        self.assertEqual(insp["bit_depth"], 16, "WAV bit depth must be 16-bit.")
        self.assertAlmostEqual(insp["duration"], 1.0, delta=0.05)

    def test_normalize_arbitrary_audio_container(self):
        """Verify arbitrary format/rate audio bytes are properly decoded and normalized to 16kHz mono WAV."""
        bio = io.BytesIO()
        rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(tone.tobytes())

        normalized = normalize_audio_to_16k_mono_wav(bio.getvalue(), target_rate=16000)
        insp = inspect_audio_wav(normalized)
        self.assertTrue(insp["is_valid"])
        self.assertEqual(insp["rate"], 16000)
        self.assertEqual(insp["channels"], 1)
        self.assertEqual(insp["bit_depth"], 16)
        self.assertAlmostEqual(insp["duration"], 1.0, delta=0.05)

    def test_transcription_quality_guard_detects_hallucinations_and_noise(self):
        """Verify the quality guard catches hallucinations and non-speech artifacts."""
        # 1. User reported hallucination
        hallucination = "बेटी, आप हँसते हो, हँसते रह जाओगे"
        res_hal = evaluate_transcription_quality(hallucination, audio_duration=4.0)
        self.assertFalse(res_hal["is_reliable"])
        self.assertIn("hallucination", res_hal["reason"].lower())

        # 2. Noise marker
        res_noise = evaluate_transcription_quality("<noise>", audio_duration=3.0)
        self.assertFalse(res_noise["is_reliable"])
        self.assertIn("inaudible", res_noise["reason"].lower())

        # 3. NO_SPEECH marker
        res_ns = evaluate_transcription_quality("NO_SPEECH", audio_duration=2.0)
        self.assertFalse(res_ns["is_reliable"])

        # 4. Implausibly short / timestamps
        res_ts = evaluate_transcription_quality("12:34", audio_duration=3.0)
        self.assertFalse(res_ts["is_reliable"])

        # 5. Legitimate Hindi/Hinglish speech
        valid_speech = "Mujhe do teen din se loose motion hai aur pet mein dard hai"
        res_val = evaluate_transcription_quality(valid_speech, audio_duration=3.5)
        self.assertTrue(res_val["is_reliable"])


if __name__ == "__main__":
    unittest.main()
