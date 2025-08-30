import streamlit as st
import os
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Model mapping dictionary
MODEL_MAPPING = {
    "Llama 8B": "llama3-8b-8192",
    "Llama 70B": "llama3-70b-8192",
    "Gemini": "gemma2-9b-it",
    "Mixtral": "mixtral-8x7b-32768",
    "DeepSeek R1": "deepseek-r1-distill-llama-70b",
    # Additional models with corrected IDs
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile", 
    "Llama Guard 4 12B": "meta-llama/llama-guard-4-12b",
    "Kimi K2": "moonshotai/kimi-k2-instruct",
    "Qwen 3 32B": "qwen/qwen3-32b",
    "Llama 4 Maverick 17B": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Llama 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
    # CodeLlama models (corrected IDs)
    "CodeLlama 7B": "llama3-groq-7b-8192-tool-use-preview",
    "CodeLlama 13B": "llama3-groq-70b-8192-tool-use-preview",
    "CodeLlama 34B": "llama-3.1-70b-versatile",
}

# Complete model descriptions for all models
MODEL_DESCRIPTIONS = {
    "DeepSeek R1": "DeepSeek R1 is a powerful LLaMA-based model optimized for efficiency and high-quality responses.",
    "Llama 8B": "Llama 8B is a mid-sized model providing balanced performance and accuracy for various AI applications.",
    "Llama 70B": "Llama 70B is an advanced model with enhanced capabilities for reasoning and detailed answers.",
    "Gemini": "Gemini (Gemma2-9B) is a fine-tuned model designed for high-quality conversational AI.",
    "Mixtral": "Mixtral 8x7B is a mixture of experts model providing exceptional performance on large-scale tasks.",
    # New model descriptions
    "Llama 3.1 8B Instant": "Llama 3.1 8B Instant is Meta's fast, efficient model with 131K context window for quick responses.",
    "Llama 3.3 70B Versatile": "Llama 3.3 70B Versatile is Meta's latest large model with enhanced reasoning and 131K context.",
    "Llama Guard 4 12B": "Llama Guard 4 12B is Meta's safety model designed to detect harmful content and ensure responsible AI.",
    "Kimi K2": "Kimi K2 is Moonshot AI's 1 trillion parameter MoE model with advanced tool use and agentic capabilities.",
    "Qwen 3 32B": "Qwen 3 32B is Alibaba's latest generation model with groundbreaking reasoning and multilingual support.",
    "Llama 4 Maverick 17B": "Llama 4 Maverick 17B is Meta's experimental 17B parameter model with mixture-of-experts architecture.",
    "Llama 4 Scout 17B": "Llama 4 Scout 17B is Meta's scout model for testing new capabilities with 16 expert modules.",
    "CodeLlama 7B": "CodeLlama 7B is a specialized model for code generation with tool use capabilities.",
    "CodeLlama 13B": "CodeLlama 13B is an advanced code generation model with enhanced programming abilities.",
    "CodeLlama 34B": "CodeLlama 34B is a large-scale model optimized for complex coding tasks and software development."
}

# Streamlit Page Configuration
st.set_page_config(page_title="All in One Chatbot", layout="centered")

# Sidebar Configuration
with st.sidebar:
    st.title("🤖 Chatbot Settings")

    selected_model = st.selectbox("Select Model", list(MODEL_MAPPING.keys()))
    model_name = MODEL_MAPPING[selected_model]
    
    st.divider()
    st.subheader("⚙️ Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, help="Controls creativity (0=focused, 1=creative)")
    max_tokens = st.slider("Max Tokens", 100, 4096, 1024, help="Maximum response length")
    top_p = st.slider("Top-P", 0.0, 1.0, 1.0, help="Nucleus sampling parameter")
    frequency_penalty = st.slider("Frequency Penalty", 0.0, 1.0, 0.0, help="Reduces repetitive responses")
    
    st.divider()
    
    # Show model info
    st.subheader("ℹ️ Model Info")
    st.info(MODEL_DESCRIPTIONS.get(selected_model, "No description available."))
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hi, I'm All in One ChatModel! I use different AI models. You've selected **{selected_model}**."}
        ]
        st.rerun()

# Main title
st.title("--MultiChat AI Chatbot--")
st.markdown(f"**Current Model:** {selected_model} | **Total Models Available:** {len(MODEL_MAPPING)}")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", 
         "content": f"Hi, I'm your All-in-One AI ChatBot! 🤖\n\nI use different AI models to help you. You've selected **{selected_model}**.\n\nHow can I assist you today?",
        }
    ]

# Display chat history with formatted response
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message["role"] == "assistant" and "caption" in message:
            st.caption(message["caption"])

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are {model_name}, a powerful AI model. 
    {model_description}
    
    Answer the question to the best of your ability, even if no additional context is provided.
    Provide the most accurate response based on the question.
    Be helpful, informative, and engaging in your responses.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# Chat input field
if user_prompt := st.chat_input("💬 Ask me anything..."):
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)
    
    try:
        # Show thinking indicator
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"🧠 {selected_model} is thinking..."):
                llm = ChatGroq(
                    groq_api_key=groq_api_key, 
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty
                )

                document_chain = create_stuff_documents_chain(llm, prompt_template)
                
                start = time.process_time()
                response = document_chain.invoke({
                    "input": user_prompt,
                    "model_name": selected_model,
                    "model_description": MODEL_DESCRIPTIONS.get(selected_model, "A powerful AI model."),
                    "context": "",
                })
                elapsed_time = time.process_time() - start
                
                bot_response = response

                print(f"[{selected_model}] Response: {bot_response}")
            
            # Store assistant response in session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "caption": f"🟡 *Generated by {selected_model}* • ⏱️ {elapsed_time:.2f}s • 🔥 Temp: {temperature}"
            })
            
            # Display the response
            st.markdown(bot_response, unsafe_allow_html=True)
            st.caption(f"🟡 *Generated by {selected_model}* • ⏱️ {elapsed_time:.2f}s • 🔥 Temp: {temperature}")
            
            # Auto-scroll to bottom
            st.rerun()

    except Exception as e:
        st.error(f"⚠️ Error with {selected_model}: {str(e)}")
        st.info("💡 Try selecting a different model or check your API key.")
        
        

