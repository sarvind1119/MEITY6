from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get('OPENAI_API_KEY'))

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know. ALWAYS give results in bullets points. '""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Ministry of Electronics and Information Technology (MEITY)...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to the document of the Ministry of Electronics and Information Technology.

    [Documents Repository1](https://drive.google.com/drive/folders/12CviwBib5xdWy3pW5trrOJxPbZFht2cn?usp=sharing)
    ''')
    [Documents Repository1](https://drive.google.com/drive/folders/1TOZlelRvNGEWt9C6rnS_oTXW4rOUZyaE?usp=share_link)
    ''')
    
    # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Shubham Bhaisare
    </div>
    ''', unsafe_allow_html=True)
    # Adding the list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>Annual Report_2022-23.pdf</li>
        <li>Annual Report_2017-18.pdf</li>
        <li>AR2016-17_English.pdf</li>
        <li>Data_Protection_Committee_Report.pdf</li>
        <li>Extension of tenure of PLI LSEM 23.09.20...</li>
        <li>FAQ_Intermediary_Rules_2021.pdf</li>
        <li>GIGW-Certificate.pdf</li>
        <li>IT_ACT_2000.pdf</li>
        <li>MeitY_Annual Report_2021-22.pdf</li>
        <li>MeitY_AR_2018-19.pdf</li>
        <li>National-Strategy-for-Artificial-Intelligen...</li>
        <li>Personal_Data_Protection_Bill_2018.pdf</li>
        <li>production_linked_incentive_scheme.pdf</li>
        <li>The Digital Personal Data Protection Bill, 2...</li>
        <li>4 Committee report on Artificial intelligence for policy making_LT.pdf</li>
        <li>Aadhaar rules and other important notifications_LT.pdf</li>
        <li>Aadhaar_Act_2016_LT.pdf</li>
        <li>Compendium on digital India_LT.pdf</li>
        <li>Digital Personal Data Protection Act 2023_LT.pdf</li>
        <li>Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021 and 2023_LT.pdf</li>
        <li>IT act amendment and circulars_LT.pdf</li>
        <li>IT act_LT.pdf</li>
        <li>IT Rules 2000_LT.pdf</li>
        <li>MeitY - Know your ministry_LT.pdf</li>
        <li>Recruitment - Hindi RR_LT.pdf</li>
        <li>Report of Task Force to suggest measures to stimulate the growth of IT ITeS and Electronics Hardware manufacturing industry_LT.pdf</li>
        <li>Report on mapping the manpower skills in the IT Hardware and Electronics Manufacturing industry_LT.pdf</li>
        <li>Research papers_LT.pdf</li>
        <li>RTI act_LT.pdf</li>
        <li>Skill Gap Analysis Report for Electronics and IT hardware Industry - Report on Human Resource and Skill Requirements in Electronics and IT hardware Sector_LT.pdf</li>
        <li>Study on semiconductor design, embedded software and services industry_LT.pdf</li>

    </ul>
    </div>
    ''', unsafe_allow_html=True)

    # Add vertical space
    st.markdown('''
    ---

    **In case of suggestions/feedback/Contributions please reach out to:**
    [NIC Training Unit @ nictu@lbsnaa.gov.in ]
    ''')

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = ask_and_get_answer(vectorstore,refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

            if "source_documents" in context:
                st.write("### Reference Documents")
                for i, doc in enumerate(context['source_documents'], start=1):
                    st.write(f"#### Document {i}")
                    st.write(f"**Page number:** {doc.metadata['page']}")
                    st.write(f"**Source file:** {doc.metadata['source']}")
                    content = doc.page_content.replace('\n', '  \n')  # Ensuring markdown line breaks
                    st.markdown(content)




with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                

          
