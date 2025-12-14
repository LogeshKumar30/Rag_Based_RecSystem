import sys
sys.path.append("..")

import sys
import os
import json
import time
import pandas as pd
import streamlit as st
# regex for demo search
import re
# from streamlit_extras.mention import mention
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMADB_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False
import json
# LangChain imports are done lazily where needed to allow demo-only runs without langchain installed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import yaml

# MUST be called first, before any other Streamlit commands
st.set_page_config(
    page_title="BigBasket Products",
    page_icon="🧺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Resolve paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Read YAML file from script directory to avoid cwd issues
config_file = os.path.join(BASE_DIR, "config.yaml")
with open(config_file, 'r') as stream:
    CONFIG = yaml.safe_load(stream)

# Number of records to retrieve
K=4

# Access your secret
api_key = os.getenv("API_KEY")

# FORCE DEMO MODE: set to True to avoid any external API calls (useful when no credits available)
DEMO_MODE = True

#############################################################################################################
#############################################################################################################

# Promp Template to be used for generating questions
# @st.cache_resource(show_spinner=False)
def PROMPT():
    prompt_template = '''
        You are a Product Recommendation Agent who gets his context from the retrieved descriptions of the products that matches best with the User's query. 
        User is a human who, as a customer, wants to buy a product from this application.
        Given below is the summary of conversation between you (AI) and the user (Human):
        Context: {chat_history}
        Now use this summary of previous conversations and the retrieved descriptions of products to answer the following question asked by the user:
        Question: {question}
        Note: 
        - Give your answer in a compreshenive manner in enumerated format.
        - Do not generate any information on your own, striclty stick to the provided data. 
        - Also, do not repeat the information that is already present in the context.
        - If, you feel there is redundant information (or) an product is being described twice, specify that as well in the response.
        - The tone of the answer should be like a polite and friendly AI Assistant.
    '''
    class SimplePrompt:
        def __init__(self, template):
            self.template = template

        def format(self, **kwargs):
            return self.template.format(**kwargs)

    return SimplePrompt(prompt_template)

def PROMPT_intent_validator():
    prompt_template_intent = """
        You are an intent identifier and comparer. 
        You will be given old_data and a new_data. 
        Your task is to identify the intents of old_data and new_data independently.
        After that you will have to compare both the intents and check if the intents have the same context or not.
        STRICTLY Reply with a lowercase yes/no.
        The following is the old_data:
        ```
        <old_data>
        ```
        The following is the new_data:
        ```
        <new_data>
        ```
        NOTE: 
        - Don't generate any additional texts, just repond with yes (or) no.
        - if `old_data' is empty, then strictly reply with 'no'
        """
    return prompt_template_intent

# Load the LLM model for inference
# @st.cache_resource(show_spinner=False)
def load_model():
    if DEMO_MODE:
        return None
    try:
        from langchain.chat_models import ChatOpenAI
        model = ChatOpenAI(
            model=CONFIG['LLM_MODEL'],
            api_key=api_key,
            base_url=CONFIG["BASE_URL"],
            max_tokens = 10000,
            # temperature = 0.7,
            # top_p = 0.9
        )
    except Exception as e:
        st.warning(f"⚠️ Could not load LLM: {str(e)[:100]}...\n\n📌 Running in DEMO MODE with mock responses.")
        model = None
    return model

llm = load_model()
# print(CONFIG["BASE_URL"])
# print(api_key)

# Memory to store the conversation history
def memory():
    if DEMO_MODE:
        # Simple in-memory demo memory that provides a single concatenated chat summary
        class DemoMemory:
            def load_memory_variables(self, inputs=None):
                msgs = st.session_state.get('messages', [])
                summary = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                class SimpleMsg:
                    def __init__(self, content):
                        self.content = content
                return {"chat_history": [SimpleMsg(summary)]}

            def clear(self):
                st.session_state['messages'] = []

        if 'demo_memory' not in st.session_state:
            st.session_state['demo_memory'] = DemoMemory()
        return st.session_state['demo_memory']

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key='answer'
        )
    return st.session_state.memory

# Wrapper for DeepInfraEmbeddings generation
class DeepInfraEmbeddings:
    def __init__(self, api_key, base_url, model=CONFIG["EMBED_MODEL"]):
        """Intialise client to access embedding model
        Args:
            api_key (str): Deep-Infra API key
            base_url (str): URL to access the embeddings
            model (str, optional): 1024 dimension embeddings. Defaults to "BAAI/bge-large-en-v1.5".
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts):
        """Converts given INPUT data to corresponding embeddings
        Args:
            texts (str): INPUT database contents as string.
        Returns:
            list: List of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )

        return [embedding.embedding for embedding in embeddings.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Retriever to retrieve the products from the database
# @st.cache_resource(show_spinner=False)
def retriever(K):
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, 'vector_stores'))

    embeddings = DeepInfraEmbeddings(
                        api_key=api_key,
                        base_url=CONFIG["BASE_URL"]
                    )

    vector_store = Chroma(
                        collection_name=CONFIG["COLLECTION_NAME"],
                        embedding_function=embeddings,  # Pass the DeepInfraEmbeddings instance
                        client=client,
                        persist_directory = os.path.join(BASE_DIR, 'vector_stores')
                    )

    retriever = vector_store.as_retriever(search_kwargs={'k':K})

    return retriever

# Chain to chain the retriever with memory
def Chain():
    global K
    if DEMO_MODE:
        return None

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever(K),
        memory=memory(),
        return_source_documents=True,
    )

    return chain

# Demo mode mock response for testing without API
def demo_search_response(user_question, response_length=None):
    """Search the local BigBasket CSV and return top-K matches in demo mode."""
    import json, re
    data_path = CONFIG.get('DATA_PATH', './data/bigBasketProducts.csv')
    # Resolve path relative to script directory (BASE_DIR defined earlier)
    if not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(BASE_DIR, data_path))
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        # fallback to simple mock if CSV missing: create a minimal sample dataset
        sample = [
            {
                'product': 'Good Day Butter Cookies',
                'brand': 'Britannia',
                'sale_price': 45.0,
                'rating': 4.3,
                'description': 'Classic buttery cookies perfect for tea-time snacks.'
            },
            {
                'product': 'Hide & Seek Fab Choco Fills',
                'brand': 'Parle',
                'sale_price': 40.0,
                'rating': 4.1,
                'description': 'Crunchy cookies filled with smooth chocolate cream.'
            },
            {
                'product': 'Oreo Original Cream Biscuits',
                'brand': 'Cadbury',
                'sale_price': 60.0,
                'rating': 4.4,
                'description': 'Chocolate-flavoured biscuits with vanilla cream filling.'
            },
            {
                'product': 'Baby Gentle Shampoo',
                'brand': 'Johnson',
                'sale_price': 150.0,
                'rating': 4.2,
                'description': 'Tear-free and mild shampoo for babies.'
            }
        ]
        df = pd.DataFrame(sample)
        st.warning(f"No local dataset found at {data_path}. Using an embedded sample dataset for demo mode. Error: {e}")

    # Normalize columns and map common alternate column names to expected columns
    alt_map = {
        'product': ['product', 'ProductName', 'product_name', 'title', 'name'],
        'brand': ['brand', 'Brand', 'manufacturer'],
        'sale_price': ['sale_price', 'saleprice', 'SalePrice', 'DiscountPrice', 'Price', 'price'],
        'rating': ['rating', 'Rating', 'avg_rating', 'review_rating'],
        'description': ['description', 'Description', 'details', 'Detail', 'Category', 'SubCategory', 'Absolute_Url']
    }

    for target_col, candidates in alt_map.items():
        if target_col not in df.columns:
            for cand in candidates:
                if cand in df.columns:
                    df[target_col] = df[cand].astype(str)
                    break
    # Ensure expected columns exist
    for col in ['product', 'brand', 'sale_price', 'rating', 'description']:
        if col not in df.columns:
            df[col] = ''

    # simple keyword matching
    tokens = [t.lower() for t in re.findall(r"\w+", user_question) if len(t) > 2]
    def score_row(r):
        text = f"{r['product']} {r['brand']} {r['description'] }".lower()
        return sum(text.count(tok) for tok in tokens)

    df['match_score'] = df.apply(score_row, axis=1)
    matches = df[df['match_score'] > 0].copy()
    if matches.empty:
        # if no keyword match, fall back to top-rated products
        candidates = df.sort_values(by='rating', ascending=False).head(K)
    else:
        candidates = matches.sort_values(by=['match_score','rating'], ascending=[False, False]).head(K)

    # Build answer and source documents
    lines = [f"Based on your query about '{user_question}', here are the recommended products:\n"]
    source_docs = []
    class MockDoc:
        def __init__(self, content):
            self.page_content = content

    # Determine verbosity preference (prefer param, else session state)
    response_length = response_length or st.session_state.get('response_length', 'Medium')

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        prod = row['product']
        brand = row['brand']
        price = row['sale_price']
        rating = row.get('rating', '')
        desc = row['description']
        # source_docs append is done after composing full message for consistency

        # construct variable verbosity
        if response_length == 'Short':
            lines.append(f"{i}. {prod} by {brand} - Price: ₹{price}\n")
        elif response_length == 'Medium':
            lines.append(f"{i}. {prod} by {brand} - Price: ₹{price} - Rating: {rating}⭐\n   Details: {desc}\n")
        elif response_length == 'Long':
            lines.append(f"{i}. {prod} by {brand} - Price: ₹{price} - Rating: {rating}⭐\n   Category: {row.get('Category','')} | Subcategory: {row.get('SubCategory','')}\n   Details: {desc}\n")
        else:  # Very long: detailed explanation + suggestions
            extra = []
            if 'Category' in row and row.get('Category'):
                extra.append(f"Category: {row.get('Category')}")
            if 'SubCategory' in row and row.get('SubCategory'):
                extra.append(f"Subcategory: {row.get('SubCategory')}")
            url_str = f" More info: {row.get('Absolute_Url','')}" if row.get('Absolute_Url', '') else ''
            # Build a small pros/cons and alternatives block with numeric parsing
            pros = []
            cons = []
            # heuristics for pros and cons based on rating/price
            try:
                rfloat = float(str(rating)) if rating not in (None, '', 'nan', 'NaN') else None
            except Exception:
                rfloat = None
            try:
                price_float = float(str(price)) if price not in (None, '') else None
            except Exception:
                price_float = None
            if rfloat is not None and rfloat >= 4.0:
                pros.append('High customer rating')
            if price_float is not None and price_float < 500:
                pros.append('Affordable price')
            if brand and isinstance(brand, str) and len(brand) > 0:
                pros.append(f'Reputable brand: {brand}')
            # simple cons heuristic
            if rfloat is not None and rfloat < 3.0:
                cons.append('Lower rating — check recent reviews')
            if price_float is not None and price_float > 1000:
                cons.append('Higher priced than alternatives')
            # Get up to two alternatives from candidates that aren't this product
            other_candidates = candidates[candidates['product'] != prod].head(2)
            alt_lines = []
            for _, orow in other_candidates.iterrows():
                alt_lines.append(f"{orow.get('product','')} by {orow.get('brand','')} (₹{orow.get('sale_price','')})")

            lines.append(f"{i}. {prod} by {brand} - Price: ₹{price} - Rating: {rating}⭐\n   {', '.join(extra)}\n   Details: {desc}\n   Why this product: This product is a good match for your query because it matches the category and has a solid rating and price. Consider how you will use it and compare features with other matches. {url_str}\n\n")
            if pros:
                lines.append("   Pros:\n")
                for p in pros:
                    lines.append(f"     - {p}\n")
            if cons:
                lines.append("   Cons:\n")
                for c in cons:
                    lines.append(f"     - {c}\n")
            # Additional suggestions for using the product
            lines.append("   Usage suggestions:\n")
            if 'cookie' in prod.lower() or 'choco' in prod.lower() or 'biscuit' in prod.lower():
                lines.append("     - Best enjoyed with tea or coffee, great for snacks and travel.\n")
            else:
                lines.append("     - Consider how you will use this product in everyday life and compare sizes/prices.\n")
            if alt_lines:
                lines.append("   Alternatives:\n")
                for a in alt_lines:
                    lines.append(f"     - {a}\n")
            lines.append("\n")
        source_docs.append(MockDoc(json.dumps({
            'product': prod,
            'brand': brand,
            'sale_price': price,
            'rating': rating,
            'description': desc
        })))

    answer = "\n".join(lines) + "\n[DEMO MODE - results from local dataset]"
    return {'answer': answer, 'source_documents': source_docs}

# Search function to search for the products
# @st.cache_data(show_spinner=False)
def search(_chain, user_question, response_length=None):
    # prefer explicit response_length or state
    response_length = response_length or st.session_state.get('response_length', 'Medium')
    if DEMO_MODE:
        return demo_search_response(user_question, response_length)
    
    intent_prompt = PROMPT_intent_validator()
    intent_prompt = intent_prompt.replace("<old_data>", memory().load_memory_variables({})['chat_history'][0].content)
    intent_prompt = intent_prompt.replace("<new_data>", user_question)

    intent_sys_message = SystemMessage(content=intent_prompt)
    intent_user_message = HumanMessage(content="Start.")
    intent_messages = [intent_sys_message, intent_user_message]
    intent_response = llm(intent_messages)

    print("INTENT_VALIDATE:", intent_response.content)
    if intent_response.content == "no":
        memory().clear()
    
    # Add verbosity instruction based on user-selected response length
    # response_length already set above (from parameter or session state)
    verbosity_map = {
        'Short': 'Be concise and keep the answer brief (short bullet points).',
        'Medium': 'Provide a standard-length response covering the main points.',
        'Long': 'Provide a detailed response with examples and short explanations.',
        'Very long': 'Provide an in-depth, comprehensive response with detailed explanations, comparisons, and recommendations.'
    }
    verbosity_instruction = verbosity_map.get(response_length, '')

    gen_prompt = PROMPT().format(question=user_question, 
                                 chat_history=memory().load_memory_variables({})['chat_history'][0].content)
    if verbosity_instruction:
        gen_prompt = gen_prompt + "\n\n" + verbosity_instruction
    try:
        res = _chain(gen_prompt)
    except Exception as e:
        st.error(e)
        res = None
    return res

#############################################################################################################
#############################################################################################################

# Initialize the app
def init():
    global K
    # Sidebar controls
    with st.sidebar:
        st.subheader('Parameters')
        K = st.slider('K', 1, 10, K, help='Sets max number of products  \nthat can be retrieved')
        st.markdown('---')
        # Response length control: Short -> Very long
        resp_len = st.selectbox('Response length', ['Short', 'Medium', 'Long', 'Very long'], index=3, help='How verbose should the assistant be?')
        st.session_state['response_length'] = resp_len
        st.markdown('**Mode:** ' + ("DEMO" if DEMO_MODE else "LIVE"))
        st.markdown('**Data:**** BigBasket products')

    # Page header with simple styling
    st.markdown(
        """
        <style>
        .title {font-size:32px; font-weight:700; margin-bottom:6px}
        .subtitle {color: #6c757d; margin-top:0; margin-bottom:12px}
        .app-container {background-color: #f8fafc; padding: 12px; border-radius: 8px}
        </style>
        <div class='app-container'>
          <div class='title'>BigBasket Products — Recommendation Assistant</div>
          <div class='subtitle'>Ask for product recommendations, comparisons, and details.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display the retrieved products
def display_data(res):
    try:
        # Safely parse source_documents that may contain json strings or dicts
        def _parse_content(c):
            try:
                if isinstance(c, (str, bytes)):
                    return json.loads(c)
                if isinstance(c, dict):
                    return c
            except Exception:
                return {}
            return {}

        srcs = [_parse_content(getattr(row, 'page_content', {})) for row in res.get('source_documents', [])]
        # Filter out empty payloads
        srcs = [s for s in srcs if s]
        df = pd.DataFrame(srcs)
    except Exception as e:
        st.error(e)
        return

    # Ensure expected columns are present so slicing doesn't raise KeyError
    expected_cols = ['product', 'brand', 'sale_price', 'rating', 'description']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''
    df1 = df[expected_cols]

    # Remove duplicates
    df1 = df1.drop_duplicates()

    st.dataframe(
        df1,
        column_config={
            "product": st.column_config.Column(
                "Product Name",
                width="medium"
            ),
            "brand": st.column_config.Column(
                "Brand",
                width="medium"
            ),
            "sale_price": st.column_config.NumberColumn(
                "Sale Price",
                help="The price of the product in USD",
                min_value=0,
                max_value=1000,
                format="₹%f",
            ),
            "rating": st.column_config.NumberColumn(
                "Rating",
                help="Rating of the product",
                format="%f ⭐",
            ),
            "description": "Description",
        },
        hide_index=True,
    )

def main():

    init()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your BigBasket Assistant — I can recommend products, compare options, and help you find the best deals. Tell me what you're looking for."}
        ]

    chain = Chain()

    # Layout: chat on the left, results on the right
    chat_col, result_col = st.columns([3, 2])

    # Chat input and messages in left column
    with chat_col:
        if "messages" not in st.session_state or not st.session_state.messages:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm your BigBasket Assistant — I can recommend products, compare options, and help you find the best deals. Tell me what you're looking for."}
            ]

        if prompt := st.chat_input("Say something"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"], unsafe_allow_html=False)

        # If last message is not from assistant, generate a new response
        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            user_prompt = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    res = search(chain, user_prompt, st.session_state.get('response_length', 'Medium'))
                    st.session_state['_last_result'] = res
                    end_time = time.time()
                    st.toast(f'Search completed in :green[{end_time - start_time:.2f}] seconds', icon='✅')
                    if res is None:
                        st.error("Something went wrong. Please try again.")
                        return

                    answer = res['answer']
                    message = {"role": "assistant", "content": answer}
                    st.session_state.messages.append(message)

                    # Stream the assistant response
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=False)
                    message_placeholder.markdown(full_response, unsafe_allow_html=False)
    # Results and product table in right column
    with result_col:
        st.markdown("**Recommendations**")
        # Show latest results if available
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            # last assistant response corresponds to last search result
            # We stored the last result in a temporary key when available
            last_res = st.session_state.get('_last_result', None)
            if last_res:
                display_data(last_res)
            else:
                st.info("No results to display yet.")

if __name__ == "__main__":
    main()

