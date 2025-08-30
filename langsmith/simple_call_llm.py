from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Pakistan?"})
print(result)