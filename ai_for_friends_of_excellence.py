import streamlit as st
import os
import re
import hashlib
import tempfile
from pathvalidate import sanitize_filename
import regex
import copy
import base64
import json
import threading
from aife_time import now_and_choices, scheduled_run
from aife_utils import RE_COMPRESS_NEWLINES, clean_yesterday_files
from aife_tools import get_prompt, get_response_format, get_tools, yusi_chat, speech_to_text, jobs, llms, resize_images, parse_pdfs, parse_txts

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.components.v1.html(
    """
    <script>
    const updateWindowHeight = () => {
        const windowHeight = window.parent.innerHeight;
        document.cookie = `window_height=${windowHeight};path=/`;
    };
    updateWindowHeight();
    window.parent.addEventListener('resize', updateWindowHeight);
    </script>
    """,
    height=0
)

window_height = int(st.context.cookies.get("window_height", 800))

st.session_state["job"] = st.query_params.get("job", st.session_state.get("job", "Text chat"))
st.session_state["llm"] = st.session_state.get("llm", jobs[st.session_state["job"]]["llms"][0])
st.session_state["chat_history"] = st.session_state.get("chat_history", [])
st.session_state["chat_history_editable"] = st.session_state.get("chat_history_editable")
st.session_state["is_chat_history_edited"] = st.session_state.get("is_chat_history_edited", False)
st.session_state["doc_category"] = st.session_state.get("doc_category", "Plain text")
st.session_state["doc_contents"] = st.session_state.get("doc_contents", [])
st.session_state["tool_results"] = st.session_state.get("tool_results", [])
st.session_state["edit_mode"] = st.session_state.get("edit_mode", False)
st.session_state["last_audio_hash"] = st.session_state.get("last_audio_hash")


def get_file_name_and_path_tuples(files):
    file_name_and_path_tuples = []
    for file in files:
        file_name = file.name
        file_path = f"uploaded-files/{sanitize_filename(os.path.splitext(file_name)[0])} {now_and_choices()}{os.path.splitext(file_name)[1]}"
        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_name_and_path_tuples.append((file_name, file_path))
        except Exception:
            return None
    return file_name_and_path_tuples


def process_documents(file_name_and_path_tuples):
    contents = {}
    category = st.session_state["doc_category"]
    pdf_name_and_path_tuples = [(file_name, file_path) for file_name, file_path in file_name_and_path_tuples if file_path.endswith(".pdf")]
    txt_name_and_path_tuples = [(file_name, file_path) for file_name, file_path in file_name_and_path_tuples if file_path.endswith(".txt")]
    if pdf_paths := [file_path for file_name, file_path in pdf_name_and_path_tuples]:
        if category == "Plain text":
            if pdf_contents := parse_pdfs(pdf_paths, 4, 1000, True):
                contents.update({(len(contents) + key): value for key, value in pdf_contents.items()})
        elif category == "Blended layout":
            if pdf_contents := parse_pdfs(pdf_paths, 4, 1000):
                contents.update({(len(contents) + key): value for key, value in pdf_contents.items()})
        elif category == "Dense visual":
            if pdf_contents := parse_pdfs(pdf_paths, 2, 500):
                contents.update({(len(contents) + key): value for key, value in pdf_contents.items()})
    if txt_paths := [file_path for file_name, file_path in txt_name_and_path_tuples]:
        if txt_contents := parse_txts(txt_paths, 5000, 1000):
            contents.update({(len(contents) + key): value for key, value in txt_contents.items()})
    if contents:
        txt_path = f"temp-data/{now_and_choices()}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(contents))
        doc_names = [file_name for file_name, file_path in pdf_name_and_path_tuples + txt_name_and_path_tuples]
        st.session_state["doc_contents"].append({"role": "user", "content": json.dumps({"doc_content": f"<doc_content>\n{'\n\n'.join(contents[key] for key in sorted(contents))}\n</doc_content>", "txt_path": f"The path of the TXT file containing the doc_content: {txt_path}"}, ensure_ascii=False), "doc_names": doc_names})


def process_images(file_name_and_path_tuples):
    image_name_and_path_tuples = [(file_name, file_path) for file_name, file_path in file_name_and_path_tuples if file_path.endswith((".jpg", ".jpeg", ".png"))]
    if image_paths := [file_path for file_name, file_path in image_name_and_path_tuples]:
        resize_images(image_paths, 1280)
    image_names = [file_name for file_name, file_path in image_name_and_path_tuples]
    return image_names, image_paths


def select_job():
    st.query_params.update({"job": st.session_state["job"]})
    st.session_state["llm"] = jobs[st.session_state["job"]]["llms"][0]


def sync_chat_history():
    pattern = r"(User:\n|AI \([^)]+\):\n)"
    segments = ["User:\n"] + [segment for segment in re.split(pattern, st.session_state["chat_history_editable"]) if segment.strip()]
    return [{"role": "user" if segments[i] == "User:\n" else "assistant", "content": segments[i + 1].strip()} for i in range(len(segments) - 1) if re.match(pattern, segments[i]) and not re.match(pattern, segments[i + 1])]


def sync_chat_history_editable():
    return "\n\n".join([f"User:\n{message['content']}" if message["role"] == "user" else f"AI ({st.session_state['llm']}):\n{message['content']}" for message in st.session_state["chat_history"]])


def is_chat_history_edited():
    st.session_state["is_chat_history_edited"] = True


def interleave_messages(messages):
    for i in reversed(range(1, len(messages))):
        last_message, current_message = messages[i - 1], messages[i]
        if last_message["role"] == current_message["role"] and isinstance(current_message["content"], str):
            last_message["content"] += "\n\n" + current_message.pop("content")
            messages.pop(i)
    return messages


def is_in_chinese(text):
    return bool(regex.search(r"\p{Han}", text))


def chat(continue_message=None, retry=0):
    selected_job = jobs[st.session_state["job"]]
    system_message = get_prompt(selected_job["system_message"])
    response_format = get_response_format(selected_job["response_format"])
    tools = get_tools(selected_job["tools"])
    messages = copy.deepcopy(st.session_state["chat_history"])
    attachments = copy.deepcopy(st.session_state["doc_contents"] + st.session_state["tool_results"])
    in_chinese = is_in_chinese(messages[0]["content"])
    for i, message in enumerate(messages):
        if "image_paths" in message:
            if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                messages[i] = {"role": "user", "content": [{"type": "text", "text": message["content"]}, *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}} for image_path in image_paths]]}
    if continue_message:
        messages.append({"role": "user", "content": continue_message})
    messages = [{"role": "system", "content": system_message}] + interleave_messages(messages[:-1] + attachments + [messages[-1]])
    selected_llm = llms[st.session_state["llm"]]
    result = yusi_chat(selected_llm["configs"], messages, response_format=response_format, tools=tools)
    if isinstance(result, str):
        st.session_state["chat_history"].append({"role": "assistant", "content": result})
        st.session_state["chat_history_editable"] = sync_chat_history_editable()
    elif isinstance(result, dict) and result["type"] == "yusi_tool_result":
        st.session_state["tool_results"].append({"role": "assistant", "content": f"{json.dumps(result['tool_results'], ensure_ascii=False)}"})
        chat(get_prompt("reply_with_tool_results" if retry < 5 else "reply_with_tool_results2", in_chinese=in_chinese), retry + 1)
    else:
        error_message = (f"我只能处理{selected_llm["context_length"]}个tokens的上下文长度。当前全部消息总长度可能超过了极限。请开启新会话，或者剪切部分历史消息，或者选择其它大语言模型继续对话，例如MiniMax，可以处理多达100万个tokens。" if in_chinese else f"I can only handle a context length of {selected_llm["context_length"]} tokens. The total length of all messages possibly exceeded my limit. Please start a new conversation thread, or cut parts of the chat history, or continue with other LLMs such as Gemini, which can process up to 1 million tokens.")
        st.session_state["chat_history"].append({"role": "assistant", "content": error_message})


def style_content_with_reasoning(content):
    if matched := re.search(r"^<think>([\s\S]*?)</think>\s*(.*)", content, re.DOTALL):
        if reasoning_content := RE_COMPRESS_NEWLINES.sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{reasoning_content}</div>{matched.group(2)}"
        return matched.group(2)
    return content


def show_messages():
    messages = st.session_state["chat_history"]
    for i, message in enumerate(messages):
        message_column, button_column = st.columns([74, 1])
        with message_column:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"User:\n{message['content']}")
                    if "image_paths" in message:
                        if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                            st.image(image_paths, width=240)
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(f"AI ({st.session_state['llm']}):\n{style_content_with_reasoning(message['content'])}", unsafe_allow_html=True)
        with button_column:
            if st.button("¯", key=f"delete_message_{i}", help="Delete this message", type="tertiary"):
                messages.pop(i)
                st.session_state["chat_history_editable"] = sync_chat_history_editable()
                st.rerun()
            st.download_button("ˇ", f"User:\n{message['content']}" if message["role"] == "user" else f"AI ({st.session_state['llm']}):\n{message['content']}", f"Message {i + 1} {now_and_choices()}.txt", "text/plain", key=f"download_message_{i}", help="Download this message", type="tertiary")


def style_content(content):
    return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{content}</div>"


def show_doc_contents():
    attachments = st.session_state["doc_contents"]
    for i, attachment in enumerate(attachments):
        message_column, button_column = st.columns([69, 1])
        with message_column:
            with st.chat_message("user"):
                st.write(f"Documents:\n{style_content(attachment['content'][:300])}... Click the buttons on the right to download or delete.", unsafe_allow_html=True)
        with button_column:
            if st.button("¯", key=f"delete_doc_content_{i}", help="Delete this attachment", type="tertiary"):
                attachments.pop(i)
                st.rerun()
            st.download_button("ˇ", f"Documents:\n{attachment['content']}", f"Attachment {i + 1} {now_and_choices()}.txt", "text/plain", key=f"download_doc_content_{i}", help="Download this attachment", type="tertiary")


def show_tool_results():
    attachments = st.session_state["tool_results"]
    for i, attachment in enumerate(attachments):
        message_column, button_column = st.columns([69, 1])
        with message_column:
            with st.chat_message("assistant"):
                st.write(f"Tool results:\n{style_content(attachment['content'][:300])}... Click the buttons on the right to download or delete.", unsafe_allow_html=True)
        with button_column:
            if st.button("¯", key=f"delete_tool_results_{i}", help="Delete this attachment", type="tertiary"):
                attachments.pop(i)
                st.rerun()
            st.download_button("ˇ", f"Tool results:\n{attachment['content']}", f"Attachment {i + 1} {now_and_choices()}.txt", "text/plain", key=f"download_tool_results_{i}", help="Download this attachment", type="tertiary")


user_message = (st.chat_input("Input a message") or "").strip()

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 18px;'><a href='https://friendsofexcellence.ai' style='text-decoration: none; color: inherit;'>优秀的朋友用的 AI for Friends of Excellence</a></h1>", unsafe_allow_html=True)

    with st.container(height=(window_height - 374) // 2, border=False):
        job_options = list(jobs.keys())
        st.radio("Job options", job_options, key="job", on_change=select_job, label_visibility="collapsed")
    with st.container(height=(window_height - 374) // 2, border=False):
        llm_options = jobs[st.session_state["job"]]["llms"]
        captions = [llms[llm]["intro"] for llm in llm_options]
        st.radio("LLM options", llm_options, key="llm", captions=captions, label_visibility="collapsed")

    st.toggle("Edit mode", key="edit_mode")

    with st.expander("Files and audios", expanded=False):
        files = st.file_uploader("Upload images or documents", type=["jpg", "jpeg", "png", "pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
        new_files = [file for file in files if file.name not in {file_name for doc_content in st.session_state["doc_contents"] for file_name in doc_content["doc_names"]} | {file_name for message in st.session_state["chat_history"] if "image_names" in message for file_name in message["image_names"]}]
        file_name_and_path_tuples = get_file_name_and_path_tuples(new_files)
        st.pills("Chose the doc's type", ["Plain text", "Blended layout", "Dense visual"], key="doc_category")

        if user_message_audio := st.audio_input("Click the mic and speak"):
            buffer = user_message_audio.getbuffer()
            current_audio_hash = hashlib.md5(buffer[:min(len(buffer), 882000)]).hexdigest()

            if current_audio_hash != st.session_state["last_audio_hash"]:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(buffer)
                try:
                    user_message = speech_to_text(f.name)
                except Exception as e:
                    st.warning(f"An exception occurred: {e}")
                finally:
                    st.session_state["last_audio_hash"] = current_audio_hash
                    os.remove(f.name)

if user_message:
    if st.session_state["is_chat_history_edited"]:
        st.session_state["chat_history"] = sync_chat_history()
        st.session_state["is_chat_history_edited"] = False

    try:
        category = jobs[st.session_state["job"]]["category"]
        if file_name_and_path_tuples:
            process_documents(file_name_and_path_tuples)
        if category == "multimodal":
                image_names, image_paths = process_images(file_name_and_path_tuples)
                if image_names and image_paths:
                    st.session_state["chat_history"].append({"role": "user", "content": user_message, "image_names": image_names, "image_paths": image_paths})
                else:
                    st.session_state["chat_history"].append({"role": "user", "content": user_message})
        elif category == "text":
            st.session_state["chat_history"].append({"role": "user", "content": user_message})
        chat()
    except Exception as e:
        st.warning(f"An exception occurred: {e}")

if st.session_state["edit_mode"]:
    st.text_area("Edit messages", height=window_height-310, key="chat_history_editable", on_change=is_chat_history_edited, label_visibility="collapsed")
else:
    show_messages()
    show_doc_contents()
    show_tool_results()

st.components.v1.html(
    """
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        let lastContent = '';
        function getLast30Characters(string) { return string.slice(-30); }
        function checkAndScroll() {
            const textArea = window.parent.document.querySelector('textarea[aria-label="Edit messages"]');
            if (textArea) {
                const textAreaValue = textArea.value;
                if (getLast30Characters(textAreaValue) !== getLast30Characters(lastContent)) {
                    textArea.scrollTop = textArea.scrollHeight;
                    lastContent = textAreaValue;
                }
            }
        }
        setTimeout(() => {
            const textArea = window.parent.document.querySelector('textarea[aria-label="Edit messages"]');
            if (textArea) textArea.scrollTop = textArea.scrollHeight;
        }, 300);
        setInterval(checkAndScroll, 500);
    });
    </script>
    """,
    height=0
)

cleanup_thread = threading.Thread(target=scheduled_run, args=(9, 0, clean_yesterday_files), daemon=True)
cleanup_thread.start()
