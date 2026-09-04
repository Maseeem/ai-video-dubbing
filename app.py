import base64
import os
import subprocess
import re
import tempfile
import time
from pathlib import Path

import streamlit as st
from google import genai


st.set_page_config(
    page_title="AI Video Dubbing",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 AI Video Dubbing")
st.write("Upload a short video and create a Hindi or Urdu dubbed version.")


def get_api_key():
    """Read the Gemini API key from Streamlit Secrets or an environment variable."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


def wait_for_file(client, uploaded_file, timeout_seconds=300):
    """Wait until a Gemini Files API upload is ready for use."""
    start = time.time()

    while uploaded_file.state and uploaded_file.state.name != "ACTIVE":
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("Gemini could not process the uploaded video.")

        if time.time() - start > timeout_seconds:
            raise TimeoutError("Gemini video processing took too long.")

        time.sleep(3)
        uploaded_file = client.files.get(name=uploaded_file.name)

    return uploaded_file


def get_video_duration(video_path):
    """Return video duration in seconds using ffprobe from imageio-ffmpeg."""
    import imageio_ffmpeg

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-i",
        str(video_path),
        "-f",
        "null",
        "-",
    ]

    # ffmpeg prints media information to stderr. We only need it to validate
    # that the file is a readable video, so a failed return code is enough.
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise ValueError("The uploaded file is not a readable video.")

    match = re.search(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise ValueError("Could not determine the video duration.")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def create_dubbed_video(input_video, translated_text, output_video):
    """
    Generate one translated voice track and replace the video's original audio.

    This is intentionally a simple MVP: the generated voice is one continuous
    narration track rather than perfectly time-aligned, speaker-matched dialogue.
    """
    import imageio_ffmpeg

    client = create_client()

    tts_response = client.interactions.create(
        model="gemini-3.1-flash-tts-preview",
        input=(
            "Read the following translated dialogue naturally and clearly. "
            "Do not add any words before or after it.\n\n"
            f"{translated_text}"
        ),
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [
                {"voice": "Kore"}
            ]
        },
    )

    if not getattr(tts_response, "output_audio", None):
        raise RuntimeError("Gemini TTS did not return audio.")

    audio_data = base64.b64decode(tts_response.output_audio.data)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        audio_file = temp_dir / "dubbed.wav"

        # Gemini TTS returns raw PCM audio at 24 kHz, mono, 16-bit.
        import wave

        with wave.open(str(audio_file), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        # Keep the original video stream and replace its audio stream.
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(audio_file),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            str(get_video_duration(input_video)),
            str(output_video),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg could not create the dubbed video.\n\n"
                + result.stderr[-2000:]
            )


def create_client():
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return genai.Client(api_key=api_key)


def translate_video(video_path, target_language):
    """
    Ask Gemini Flash to understand the video's spoken dialogue and translate it.

    We deliberately request plain translated dialogue so the result can be
    passed directly to the TTS model.
    """
    client = create_client()

    uploaded_file = client.files.upload(file=str(video_path))
    uploaded_file = wait_for_file(client, uploaded_file)

    prompt = f"""
Watch and understand the uploaded video.

Task:
1. Identify the spoken dialogue in the video.
2. Ignore background music, sound effects, and non-spoken sounds.
3. Translate the dialogue into {target_language}.
4. Preserve the meaning, emotion, and conversational style as naturally as possible.
5. Return ONLY the translated dialogue as plain text.
6. Do not add explanations, speaker labels, timestamps, headings, markdown, or notes.

If there is very little or no spoken dialogue, say exactly:
NO_DIALOGUE
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[uploaded_file, prompt],
    )

    text = (response.text or "").strip()

    if not text:
        raise RuntimeError("Gemini returned an empty translation.")

    if text == "NO_DIALOGUE":
        raise RuntimeError("No spoken dialogue was detected in the video.")

    return text


# -------------------------
# Streamlit UI
# -------------------------

uploaded_video = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="For this beginner MVP, use a short clip around 5–10 minutes.",
)

target_language = st.selectbox(
    "Choose dubbing language",
    ["Hindi", "Urdu"],
)

if uploaded_video:
    st.video(uploaded_video)

    if st.button("🎙️ Create Dubbed Video", type="primary"):
        api_key = get_api_key()

        if not api_key:
            st.error(
                "Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets."
            )
            st.stop()

        # Keep the MVP intentionally conservative for Streamlit Community Cloud.
        max_size_mb = 200
        file_size_mb = uploaded_video.size / (1024 * 1024)

        if file_size_mb > max_size_mb:
            st.error(
                f"Please use a video smaller than {max_size_mb} MB. "
                f"Your file is about {file_size_mb:.1f} MB."
            )
            st.stop()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / uploaded_video.name
            output_path = temp_dir / "dubbed_video.mp4"

            input_path.write_bytes(uploaded_video.getbuffer())

            try:
                # Validate that ffmpeg can read the video.
                get_video_duration(input_path)

                with st.status("Creating your dubbed video...", expanded=True) as status:
                    st.write("1/3 Uploading the video to Gemini...")
                    translated_text = translate_video(input_path, target_language)

                    st.write("2/3 Translating the dialogue...")
                    # Translation happens inside translate_video.

                    st.write("3/3 Generating dubbed speech and combining it with the video...")
                    create_dubbed_video(
                        input_path,
                        translated_text,
                        output_path,
                    )

                    status.update(
                        label="Dubbed video is ready!",
                        state="complete",
                        expanded=False,
                    )

                st.subheader("Preview")
                st.video(str(output_path))

                st.download_button(
                    "⬇️ Download Dubbed Video",
                    data=output_path.read_bytes(),
                    file_name=f"dubbed_{target_language.lower()}.mp4",
                    mime="video/mp4",
                )

                with st.expander("View translated dialogue"):
                    st.text_area(
                        "Translation",
                        translated_text,
                        height=220,
                    )

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.info(
                    "Tip: Start with a short MP4 clip and make sure "
                    "GEMINI_API_KEY is correctly configured."
                )

st.divider()
st.caption(
    "MVP note: This version creates a single continuous dubbed voice track. "
    "It does not yet perform professional lip-sync, exact dialogue timing, "
    "voice cloning, or per-speaker voice matching."
)
