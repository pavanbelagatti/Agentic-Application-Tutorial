from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_pitch(insights):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = PromptTemplate.from_template(
        "Using the insights below, generate a pitch outline:\n\n{insights}\n\n"
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(insights)