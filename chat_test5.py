import os
import sys
import time  # 時間計測用
import numpy as np
import pyaudio
from dotenv import load_dotenv

# Whisper
from faster_whisper import WhisperModel

# LangChain + OpenAI
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# langgraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Unitree SDK
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from actions import Go2Action
    UNITREE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Unitree SDK Import Error: {e}")
    print("[WARN] ロボット制御は無効化されます。")
    UNITREE_AVAILABLE = False
    Go2Action = None

# === gTTS + playsoundによるTTSライブラリ ===
from gtts import gTTS
from playsound import playsound

import tempfile  # 一時ファイル生成のために使用

# ======================================
# 1. .env ファイルからAPIキーなどを読み込む
# ======================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY が設定されていません。")

# ======================================
# 2. Whisperモデルの準備
# ======================================
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# ======================================
# 3. PyAudioでマイク設定
# ======================================
CHUNK = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("[INFO] Listening... (Press Ctrl+C to stop)")

# ======================================
# 4. ロボットアクション (Unitree)
# ======================================
if UNITREE_AVAILABLE:
    try:
        ChannelFactoryInitialize(0, "enp0s25")
        action = Go2Action()
    except Exception as e:
        print(f"[WARN] Unitree SDK 初期化失敗: {e}")
        UNITREE_AVAILABLE = False
        action = None
else:
    action = None

@tool
def StandUp():
    """立ち上がる"""
    if action:
        action.StandUp()
    else:
        print("[INFO] Robot action is unavailable (StandUp).")

@tool
def SitDown():
    """座る"""
    if action:
        action.SitDown()
    else:
        print("[INFO] Robot action is unavailable (SitDown).")

@tool
def Stretch():
    """ストレッチ"""
    if action:
        action.Stretch()
    else:
        print("[INFO] Robot action is unavailable (Stretch).")

@tool
def Dance():
    """ダンス"""
    if action:
        action.Dance()
    else:
        print("[INFO] Robot action is unavailable (Dance).")

@tool
def FrontJump():
    """前方にジャンプ"""
    if action:
        action.FrontJunmp()
    else:
        print("[INFO] Robot action is unavailable (FrontJump).")

@tool
def Heart():
    """ハートを描く"""
    if action:
        action.Heart()
    else:
        print("[INFO] Robot action is unavailable (Heart).")

@tool
def FrontFlip():
    """バク転"""
    if action:
        action.FrontFlip()
    else:
        print("[INFO] Robot action is unavailable (FrontFlip).")

@tool
def Move(x: float, y: float, z: float):
    """前方にx(m)、右にy(m)、z(rad)半時計回りに回転"""
    if action:
        action.Move(x, y, z)
    else:
        print("[INFO] Robot action is unavailable (Move).")

tools = [StandUp, SitDown, Stretch, Dance, FrontJump, Heart, FrontFlip]

# ======================================
# 5. LLMエージェントを構築
# ======================================
def create_tool_agent(model, tools):
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        messages = state['messages']
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    tool_node = ToolNode(tools)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

llm_model = ChatOpenAI(
    model="gpt-4o-mini",  # 実際のモデル名に合わせて修正
    temperature=0,
    openai_api_key=OPENAI_API_KEY
).bind_tools(tools)

app = create_tool_agent(llm_model, tools)

# ======================================
# 6. 時間計測用の構造と表示関数 (任意)
# ======================================
def print_time_log(tlog: dict):
    print("\n--- [TIME REPORT] ---------------------------")
    chunk_times = tlog["chunk_capture_times"]
    if chunk_times:
        for i, ct in enumerate(chunk_times, start=1):
            print(f"  - Chunk{i} capture time: {ct:.4f} sec")
        total_capture = sum(chunk_times)
        print(f"  => Sum of {len(chunk_times)} chunks: {total_capture:.4f} sec")

    print(f"  - Time from first chunk capture to Whisper input: {tlog['time_to_whisper_input']:.4f} sec")
    print(f"  - Combine chunks: {tlog['combine_audio']:.4f} sec")
    print(f"  - Convert to array: {tlog['convert_to_array']:.4f} sec")
    print(f"  - Whisper transcription: {tlog['whisper_time']:.4f} sec")
    print(f"  - LLM process: {tlog['llm_time']:.4f} sec")
    print(f"  - TTS generation: {tlog['tts_time']:.4f} sec")
    print("--------------------------------------------\n")


def reset_time_log():
    return {
        "chunk_capture_times": [],
        "time_to_whisper_input": 0.0,
        "combine_audio": 0.0,
        "convert_to_array": 0.0,
        "whisper_time": 0.0,
        "llm_time": 0.0,
        "tts_time": 0.0
    }

# ======================================
# 7. メインループ: 音声→Whisper→LLM→ロボット + gTTS で音声出力
# ======================================
try:
    audio_buffer = []
    time_log = reset_time_log()

    capture_block_start = None  # 5チャンクまとめて録音スタート時刻
    last_transcribed_text = ""

    while True:
        if capture_block_start is None:
            capture_block_start = time.time()

        # --- Audioキャプチャ (1 chunk)
        c_start = time.time()
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        c_end = time.time()
        capture_time = c_end - c_start
        time_log["chunk_capture_times"].append(capture_time)

        # バッファに追加
        audio_buffer.append(audio_data)

        # 5チャンク貯めたら処理
        if len(audio_buffer) >= 5:
            block_capture_end = time.time()
            time_log["time_to_whisper_input"] = block_capture_end - capture_block_start

            # 次の5チャンク計測開始時刻をリセット
            capture_block_start = None

            # 結合
            combine_start = time.time()
            audio_chunk = b''.join(audio_buffer)
            audio_buffer = []
            combine_end = time.time()
            time_log["combine_audio"] = combine_end - combine_start

            # NumPy配列へ
            convert_start = time.time()
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            convert_end = time.time()
            time_log["convert_to_array"] = convert_end - convert_start

            # --- (オプション) 無音判定 ---
            amplitude_threshold = 0.003
            avg_amplitude = np.abs(audio_array).mean()
            if avg_amplitude < amplitude_threshold:
                print(f"[INFO] Skipping due to low amplitude ({avg_amplitude:.6f}).")
                print_time_log(time_log)
                time_log = reset_time_log()
                continue

            # Whisper で認識
            transcribe_start = time.time()
            segments, _ = whisper_model.transcribe(
                audio_array,
                beam_size=5,
                language="ja",
                task="transcribe"
            )
            transcribe_end = time.time()
            time_log["whisper_time"] = transcribe_end - transcribe_start

            transcribed_text = " ".join(segment.text for segment in segments).strip()
            print(f"\n[INFO] Transcribed text: {transcribed_text}")

            # 空文字や短い場合はスキップ
            if not transcribed_text or len(transcribed_text) < 2:
                print("[INFO] Skipping empty or too-short transcription.")
                print_time_log(time_log)
                time_log = reset_time_log()
                continue

            # (オプション) 前回と同じならスキップ
            # if transcribed_text == last_transcribed_text:
            #     print("[INFO] Skipping duplicated transcription.")
            #     print_time_log(time_log)
            #     time_log = reset_time_log()
            #     continue

            # === 録音を止めてから LLM & TTS 実行 ===
            stream.stop_stream()

            # LLM に渡す
            llm_start = time.time()
            final_state = app.invoke(
                {"messages": [HumanMessage(content=transcribed_text)]},
                config={"configurable": {"thread_id": 42}}
            )
            llm_end = time.time()
            time_log["llm_time"] = llm_end - llm_start

            response = final_state["messages"][-1].content
            print(f"[LLM] {response}")

            # === gTTS で音声出力 ===
            tts_start = time.time()
            # 一時ファイルにMP3を書き出して playsound で再生
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                mp3_filename = tmp_mp3.name
            
            # 日本語の場合 lang="ja"
            tts = gTTS(text=response, lang="ja")
            tts.save(mp3_filename)

            # 再生
            playsound(mp3_filename)
            # 一時ファイル削除
            os.remove(mp3_filename)

            tts_end = time.time()
            time_log["tts_time"] = tts_end - tts_start

            # 音声再生が終わったのでストリーム再開
            stream.start_stream()

            # ログ表示
            print_time_log(time_log)
            time_log = reset_time_log()

            last_transcribed_text = transcribed_text

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

