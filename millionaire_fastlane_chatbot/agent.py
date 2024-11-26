from millionaire_fastlane_chatbot.main import index
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from millionaire_fastlane_chatbot.configuration import llm
from millionaire_fastlane_chatbot.utils import read_pdf_files, chunk_data, create_embeddings, create_vector_store, initialize_llm, initialize_qa_chain, chatbot
from langgraph.checkpoint.memory import MemorySaver


retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 2})
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_book_info",
        "Search and return information about 'The Millionaire Fastlane' book. by Mj Demarco"
    )

tools = [retriever_tool]

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

### Edges


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
  

    # LLM with tool and validation
    llm_with_tool = llm.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    
    response = llm.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import streamlit as st

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
memory = MemorySaver()

# Compile
graph = workflow.compile(checkpointer=memory)

import pprint

# inputs = {
#     "messages": [
#         ("user", "who is the author "),
#     ]
# }
# for output in graph.stream(inputs):
#     for key, value in output.items():
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint("---")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#     pprint.pprint("\n---\n")
documents = chunk_data(read_pdf_files("document/"))

# Create embeddings and vector store
embeddings = create_embeddings()
index = create_vector_store(documents, embeddings)

# Initialize LLM and QA chain
llm = initialize_llm()
chain = initialize_qa_chain(llm)


st.title("Chat with the book 'The Millionaire Fastlane'")
st.write("This is a Retrieval-Augmented Generation (RAG) app. You can ask anything about the book 'The Millionaire Fastlane'. The app will provide responses based on the contents of the book.")

# Initialize chat history
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
# Accept user input
if prompt := st.chat_input("What is?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response generation
    inputs = {"messages": [("user", prompt)]}
    output_found = False  # Flag to track if output is found

    # Stream through the graph and log output
    for output in graph.stream(inputs):
        # Log the complete output for debugging
        # pprint.pprint(output)
        
        # Check for keys that contain messages and update accordingly
        response_content = None
        
        for key in ["agent", "retrieve", "generate"]:
            if key in output and "messages" in output[key]:
                response_content = output[key]["messages"][-1]
                
                # Display the action message (Rewriting, Generating, or Grading)
                if key == "retrieve":
                    # Display retrieved content as well
                    st.chat_message("assistant").markdown("🔄 **Retrieving...**")
                    # retrieved_content = response_content["content"] if isinstance(response_content, dict) else response_content
                    with st.chat_message("assistant"):
                        st.markdown(f"Content Retrieved")
                elif key == "generate":
                    # Display generated content
                    st.chat_message("assistant").markdown("🚀 **Generating...**")
                    generated_content = response_content["content"] if isinstance(response_content, dict) else response_content
                    with st.chat_message("assistant"):
                        st.markdown(generated_content)
                
                break

    #     # If response_content is found, display it
    #     if response_content:
    #         response = response_content["content"] if isinstance(response_content, dict) else response_content
    #         with st.chat_message("assistant"):
    #             st.markdown(response)
    #             st.session_state.messages.append({"role": "assistant", "content": response})
    #         output_found = True
    #     else:
    #         # Handle the case where 'messages' key is missing
    #         st.error("Error: 'messages' key not found in the output.")
    #         st.write("Complete output for debugging:")
    #         st.json(output)
    
    # # If no output was found, provide a default response
    # if not output_found:
    #     st.write("No valid response generated. Please try again.")