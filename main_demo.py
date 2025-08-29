# main_app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import av
import requests
import time
import os
import queue
from dotenv import load_dotenv
from typing import List
import io
import wave

# Import CrewAI components
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# --- Helper Classes and Functions (No changes here) ---

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = queue.Queue()
        self.last_error = None
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            resampled_frames = self.resampler.resample(frame)
            for resampled_frame in resampled_frames:
                audio_bytes = resampled_frame.to_ndarray().tobytes()
                self.audio_buffer.put(audio_bytes)
        except Exception as e:
            self.last_error = str(e)
        return frame
        
    def get_audio_bytes(self):
        audio_segments = []
        while not self.audio_buffer.empty():
            audio_segments.append(self.audio_buffer.get())
        if not audio_segments:
            return None
        return b"".join(audio_segments)

def pcm_to_wav_in_memory(audio_bytes, sample_rate=16000):
    if not audio_bytes:
        return None
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return wav_buffer.getvalue()

# --- AssemblyAI and CrewAI Functions (No changes here) ---

def transcribe_with_assemblyai(api_key, audio_file_path):
    """Transcribes audio using AssemblyAI API."""
    headers = {'authorization': api_key}
    
    # 1. Upload the audio file
    try:
        with open(audio_file_path, 'rb') as f:
            upload_response = requests.post(
                'https://api.assemblyai.com/v2/upload',
                headers=headers,
                data=f
            )
        upload_response.raise_for_status()
        upload_url = upload_response.json()['upload_url']
    except requests.exceptions.RequestException as e:
        error_details = ""
        if e.response is not None:
            try:
                error_details = e.response.json().get('error', e.response.text)
            except:
                error_details = e.response.text
        st.error(f"Error uploading file to AssemblyAI: {e}. Details: {error_details}")
        return None
    except KeyError:
        st.error(f"Failed to get upload URL from AssemblyAI. Response: {upload_response.text}")
        return None

    # 2. Request transcription
    json_data = {
        'audio_url': upload_url,
        'speaker_labels': True,
        'punctuate': True,
        'format_text': True,
        'language_detection': True,
        'disfluencies': True
    }
    try:
        transcript_response = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            json=json_data,
            headers=headers
        )
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json()['id']
    except requests.exceptions.RequestException as e:
        st.error(f"Error requesting transcription from AssemblyAI: {e}")
        return None
    except KeyError:
        st.error(f"Failed to get transcript ID. Response: {transcript_response.text}")
        return None

    # 3. Poll for the result
    polling_endpoint = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
    while True:
        try:
            polling_response = requests.get(polling_endpoint, headers=headers)
            polling_response.raise_for_status()
            polling_result = polling_response.json()

            if polling_result['status'] == 'completed':
                return {
                    "text": polling_result.get('text'),
                    "language_code": polling_result.get('language_code', 'en')
                }
            elif polling_result['status'] == 'error':
                st.error(f"Transcription failed: {polling_result['error']}")
                return None
            
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            st.error(f"Error polling for transcription result: {e}")
            return None

def process_answer_with_crewai(openai_api_key, question, answer, language_code):
    """Processes the transcribed answer using CrewAI agents."""
    if not answer:
        return "No answer was provided to process."
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0.2)
        translator_agent = Agent(role='Expert Language Translator', goal=f'Translate the given text accurately from {language_code} to English. If already English (en), return original text.', backstory='Skilled translator ensuring fluent, accurate translations.', verbose=False, llm=llm, allow_delegation=False)
        analyzer_agent = Agent(role='Answer Relevance Analyzer', goal='Analyze the translated ENGLISH text for relevance to the question and identify key points.', backstory='Expert in linguistic analysis skilled at evaluating relevance.', verbose=False, llm=llm, allow_delegation=False)
        summarizer_agent = Agent(role='Concise Summarizer', goal='Summarize the key points of the ENGLISH answer into a clear, two-sentence format.', backstory='Professional editor talented at distilling information.', verbose=False, llm=llm, allow_delegation=False)
        translation_task = Task(description=f"Source language is '{language_code}'. Translate to English: '{answer}'. If 'en', output original text.", expected_output="The accurate English translation.", agent=translator_agent)
        analysis_task = Task(description=f"Analyze this ENGLISH answer for the question: '{question}'. Is it relevant? Extract main points.", expected_output="Key points and a conclusion on relevance.", agent=analyzer_agent, context=[translation_task])
        summary_task = Task(description="Based on the analysis, create a concise, two-sentence summary of the user's response.", expected_output="A final, polished summary.", agent=summarizer_agent, context=[analysis_task])
        qa_crew = Crew(agents=[translator_agent, analyzer_agent, summarizer_agent], tasks=[translation_task, analysis_task, summary_task], process=Process.sequential)
        return qa_crew.kickoff()
    except Exception as e:
        st.error(f"An error occurred during CrewAI processing: {e}")
        return "Could not process the answer due to an error."

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Audio Q&A with AI Analysis")

st.title("🎙️ Audio Q&A with AI Analysis")
st.markdown("Select a question from the sidebar to begin. Answer using your voice, confirm the recording, and see the AI-powered analysis.")

# --- API Key Management & Questions ---
load_dotenv()
QUESTIONS = [
    "How do you ensure safety standards are strictly followed on a busy construction site?",
    "Describe a time you dealt with an unexpected project delay. How did you manage the project timeline?",
    "What is your process for coordinating with subcontractors, architects, and clients?",
    "Tell me about a complex construction project you managed. What were the key challenges?",
]

# --- Sidebar and State Initialization ---
with st.sidebar:
    st.header("🔑 API Configuration")
    assemblyai_api_key = st.text_input("AI API Key", type="password", value=os.getenv("ASSEMBLYAI_API_KEY") or "")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    st.markdown("---")
    
# Initialize session state for non-linear flow
if 'answers' not in st.session_state:
    st.session_state.answers = {}
    st.session_state.selected_question_index = 0
    st.session_state.audio_bytes = None
    st.session_state.audio_file_path = None
    st.session_state.transcription = None
    st.session_state.language_code = None
    st.session_state.crew_result = None
    st.session_state.recording_started = False
    st.session_state.audio_processor = AudioRecorder()

# Sidebar for question navigation
with st.sidebar:
    st.header("📋 Questions")
    st.write("Select a question to answer.")
    for i, q in enumerate(QUESTIONS):
        label = f"✅ {q}" if i in st.session_state.answers else q
        if st.button(label, key=f"q_button_{i}"):
            st.session_state.selected_question_index = i
            st.session_state.audio_bytes = None
            st.session_state.audio_file_path = None
            st.session_state.transcription = None
            st.session_state.language_code = None
            st.session_state.crew_result = None
            
            st.rerun()

# --- Main Application Logic ---

# Check if all questions are answered to show the final summary
if len(st.session_state.answers) == len(QUESTIONS):
    st.header("✅ All Questions Answered!")
    st.balloons()
    st.markdown("Here is a summary of your responses:")

    for i, data in sorted(st.session_state.answers.items()):
        with st.expander(f"**Question {i+1}: {data['question']}**"):
            st.markdown(f"**Your Transcription ({data.get('language_code', 'N/A')}):**")
            st.write(data['transcription'])
            st.markdown("**AI Analysis (English):**")
            st.info(data['ai_analysis'])
            st.markdown("**Your Recording:**")
            st.audio(data['audio_file'])

    for i in range(len(QUESTIONS)):
        file_path = f"temp_audio_{i}.wav"
        if os.path.exists(file_path):
            os.remove(file_path)

# Main interface for the selected question
else:
    idx = st.session_state.selected_question_index
    current_question = QUESTIONS[idx]

    if idx in st.session_state.answers:
        st.success(f"You have already answered Question {idx + 1}. Please select another from the sidebar.")
    else:
        st.header(f"Answering Question {idx + 1}/{len(QUESTIONS)}")
        st.subheader(current_question)

        if not assemblyai_api_key or not openai_api_key:
            st.warning("Please enter your API keys in the sidebar to proceed.")
            st.stop()
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Step 1: Record Your Answer")
            st.info("Click **START** to begin recording. Click **STOP** when you are finished.")
            
            audio_processor = st.session_state.audio_processor

            webrtc_ctx = webrtc_streamer(
                key=f"audio-recorder-{idx}",
                  mode=WebRtcMode.SENDONLY, 
                  audio_processor_factory=lambda: audio_processor,
                    media_stream_constraints={"video": False, "audio": True})
            
            if webrtc_ctx.state.playing:
                if not st.session_state.recording_started:
                    st.session_state.recording_started = True
                    st.session_state.audio_processor.audio_buffer = queue.Queue()
                st.markdown("🔴 **Recording...**")

            elif not webrtc_ctx.state.playing and st.session_state.recording_started:
                st.session_state.recording_started = False
                if audio_processor.last_error:
                    st.error(f"An error occurred during recording: {audio_processor.last_error}")
                    audio_processor.last_error = None
                else:
                    
                    raw_audio_bytes = st.session_state.audio_processor.get_audio_bytes()
                if raw_audio_bytes and len(raw_audio_bytes) > 16000:
                    wav_bytes = pcm_to_wav_in_memory(raw_audio_bytes)
                    
                    file_path = f"temp_audio_{idx}.wav"
                    with open(file_path, "wb") as f:
                        f.write(wav_bytes)

                    st.session_state.audio_file_path = file_path
                    st.session_state.audio_bytes = wav_bytes
                    st.rerun()
                else:
                    st.warning("Recording was too short. Please try again.")

        with col2:
            if st.session_state.audio_bytes:
                st.markdown("#### Step 2: Confirm & Transcribe")
                st.audio(st.session_state.audio_bytes, format="audio/wav")
                
                confirm_col, rerecord_col = st.columns(2)

                if confirm_col.button("✅ Confirm and Transcribe", key=f"confirm_{idx}"):
                    with st.spinner("Transcribing your audio..."):
                        result = transcribe_with_assemblyai(assemblyai_api_key, st.session_state.audio_file_path)
                        
                        if result and isinstance(result, dict):
                            st.session_state.transcription = result.get("text")
                            st.session_state.language_code = result.get("language_code", "en")
                            
                            if st.session_state.transcription is not None:
                                st.success(f"Transcription complete! Language detected: {st.session_state.language_code}")
                            else:
                                st.error("Transcription succeeded but returned no text. The audio may have been silent.")
                                st.session_state.language_code = None
                        else:
                            st.session_state.transcription = None
                            st.session_state.language_code = None
                            st.error("Transcription failed. Please check the logs above for details and try re-recording.")
                if rerecord_col.button("🔄 Re-record", key=f"rerecord_{idx}"):
                    st.session_state.audio_bytes = None
                    st.session_state.audio_file_path = None
                    st.session_state.transcription = None
                    st.session_state.language_code = None
                    st.session_state.crew_result = None
                    st.rerun()
        if st.session_state.transcription:
            st.markdown("---")
            st.markdown("#### Step 3: Review and Process")

            st.text_area(f"Your Transcribed Answer ({st.session_state.language_code}):", value=st.session_state.transcription, height=150)
            
            if st.button("🤖 Process with AI Agents", key=f"process_{idx}"):
                with st.spinner("AI agents are analyzing your answer..."):
                    crew_result = process_answer_with_crewai(
                    openai_api_key, 
                    current_question, 
                    st.session_state.transcription,
                    st.session_state.language_code
                )
                st.session_state.crew_result = crew_result
        
        if st.session_state.crew_result:
            st.markdown("#### AI Analysis & Summary (in English)")
            st.info(st.session_state.crew_result)
            
            if st.button("💾 Save Answer", key=f"save_{idx}"):
                st.session_state.answers[idx] = {
                    "question": current_question,
                    "audio_file": st.session_state.audio_file_path,
                    "transcription": st.session_state.transcription,
                    "language_code": st.session_state.language_code,
                    "ai_analysis": st.session_state.crew_result
                }
                st.session_state.audio_bytes = None
                st.session_state.audio_file_path = None
                st.session_state.transcription = None
                st.session_state.language_code = None
                st.session_state.crew_result = None
                st.rerun()