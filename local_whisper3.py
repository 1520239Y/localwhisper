import os
import sys
import time
import numpy as np
import pyaudio
from dotenv import load_dotenv

# Whisper (faster_whisper)
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

# (オプション) Unitree SDK
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from actions import Go2Action
    UNITREE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Unitree SDK Import Error: {e}")
    print("[WARN] ロボット機能は無効化されます。")
    UNITREE_AVAILABLE = False
    Go2Action = None

# gTTS + playsound
from gtts import gTTS
from playsound import playsound
import tempfile

# ======================================
# 1. .env ファイルからAPIキーを読み込む
# ======================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY が設定されていません。")

# ======================================
# 2. Whisperモデルの準備 (faster_whisper)
# ======================================
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# ======================================
# 3. PyAudioでストリーミング録音
# ======================================
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024  # 1フレームあたり1024サンプル ≈ 64ms

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)
print("[INFO] Streaming start. (Press Ctrl+C to stop)")

# ======================================
# 4. ロボットアクション (Unitree) - オプション
# ======================================
if UNITREE_AVAILABLE:
    try:
        ChannelFactoryInitialize(0, "enp0s8")
        action = Go2Action()
        print("[INFO] ロボットSDK初期化成功")
    except Exception as e:
        print(f"[WARN] ロボットSDK初期化失敗: {e}")
        print("[WARN] ロボット機能は無効化されます。")
        UNITREE_AVAILABLE = False
        action = None
else:
    action = None

@tool
def StandUp():
    """ロボットを立ち上がらせるアクションを実行します。"""
    if action:
        action.StandUp()
    else:
        print("[INFO] Robot action is unavailable (StandUp).")

@tool
def SitDown():
    """ロボットを座らせるアクションを実行します。"""
    if action:
        action.SitDown()
    else:
        print("[INFO] Robot action is unavailable (SitDown).")

tools = [StandUp, SitDown]

# ======================================
# 5. LLMエージェント構築 (langgraph)
# ======================================
def create_tool_agent(model, tools):
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        messages = state["messages"]
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
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
).bind_tools(tools)

app = create_tool_agent(llm_model, tools)

# ======================================
# 6. 時間計測用構造 & 表示関数
# ======================================
def reset_time_log():
    return {
        "capture_start": None,
        "capture_end": None,
        "whisper_time": 0.0,
        "llm_time": 0.0,
        "tts_gen_time": 0.0,
        "tts_play_time": 0.0,
    }

def print_time_log(tlog: dict):
    print("\n--- [TIME REPORT] -----------------")
    # 録音時間
    if tlog["capture_start"] and tlog["capture_end"]:
        record_dur = tlog["capture_end"] - tlog["capture_start"]
        print(f"  - 録音時間(話し始め～終わり)   : {record_dur:.3f} sec")
    print(f"  - Whisper推論時間              : {tlog['whisper_time']:.3f} sec")
    print(f"  - LLM解析時間                  : {tlog['llm_time']:.3f} sec")
    print(f"  - TTS生成時間                  : {tlog['tts_gen_time']:.3f} sec")
    print(f"  - TTS再生時間                  : {tlog['tts_play_time']:.3f} sec")
    print("----------------------------------\n")

# ======================================
# 7. ストリーミング入力: 無音判定で区切る
# ======================================
try:
    time_log = reset_time_log()
    audio_buffer = bytearray()

    SILENCE_THRESHOLD = 700
    MAX_SILENCE_FRAMES = 15
    silence_count = 0
    capturing = False

    while True:
        # 1フレーム分の音声を取得
        frame = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)

        # numpy配列に変換して音量を測定
        frame_int16 = np.frombuffer(frame, dtype=np.int16)
        amplitude = np.mean(np.abs(frame_int16))

        # たまにDEBUGログを表示
        if np.random.rand() < 0.03:
            print(f"[DEBUG] amplitude={amplitude:.1f}")

        is_silent = (amplitude < SILENCE_THRESHOLD)

        # 「まだ喋っていない」→「喋り始めを検知」
        if not capturing:
            if not is_silent:
                capturing = True
                time_log["capture_start"] = time.perf_counter()
                audio_buffer = bytearray(frame)
                silence_count = 0
        else:
            # 既に録音中: フレームをバッファに追加
            audio_buffer.extend(frame)
            if is_silent:
                silence_count += 1
            else:
                silence_count = 0

            # 一定フレーム以上無音が続けば録音終了
            if silence_count >= MAX_SILENCE_FRAMES:
                time_log["capture_end"] = time.perf_counter()
                capturing = False
                silence_count = 0

                # === ここから処理開始 ===
                audio_data = bytes(audio_buffer)
                audio_buffer = bytearray()

                # 1) Whisper推論
                float_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)/32768.0
                w_start = time.perf_counter()
                segments, _ = whisper_model.transcribe(
                    float_array,
                    beam_size=5,
                    language="ja",
                    task="transcribe"
                )
                w_end = time.perf_counter()
                time_log["whisper_time"] = w_end - w_start

                transcribed_text = " ".join(s.text for s in segments).strip()
                print(f"[INFO] Transcribed text: {transcribed_text}")

                if not transcribed_text:
                    # 音声が認識できなかった
                    print_time_log(time_log)
                    time_log = reset_time_log()
                    continue

                # 2) LLM解析
                stream.stop_stream()
                llm_start = time.perf_counter()
                final_state = app.invoke(
                    {"messages": [HumanMessage(content=transcribed_text)]},
                    config={"configurable": {"thread_id": 42}}
                )
                llm_end = time.perf_counter()
                time_log["llm_time"] = llm_end - llm_start

                llm_response = final_state["messages"][-1].content
                print(f"[LLM] {llm_response}")

                # 3) TTS生成 + 再生 (分割して計測)
                # (a) TTS生成
                tts_gen_start = time.perf_counter()
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                    mp3_filename = tmp_mp3.name
                tts = gTTS(text=llm_response, lang="ja")
                tts.save(mp3_filename)
                tts_gen_end = time.perf_counter()
                time_log["tts_gen_time"] = tts_gen_end - tts_gen_start

                # (b) TTS再生
                tts_play_start = time.perf_counter()
                playsound(mp3_filename)
                os.remove(mp3_filename)
                tts_play_end = time.perf_counter()
                time_log["tts_play_time"] = tts_play_end - tts_play_start

                # 録音再開
                stream.start_stream()

                # 時間計測を表示 & リセット
                print_time_log(time_log)
                time_log = reset_time_log()

except KeyboardInterrupt:
    print("\n[INFO] ユーザーにより停止されました。")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
