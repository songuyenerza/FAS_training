from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

import time

class SCHEMA_OUTPUT(BaseModel):
    """Information about SCHEMA_OUTPUT."""

    score: Optional[float] = Field(..., description="score")
    nc: Optional[int] = Field(
        ..., description="number of NC"
    )
    report: Optional[str] = Field(..., description="name of REPORT")

    url: Optional[str] = Field(..., description="url link of DETAIL")



from langchain_core.prompts import ChatPromptTemplate

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality.
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)
# We will be using tool calling mode, which
# requires a tool calling capable model.
llm = ChatOpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="token-abc123",
    model="Qwen/Qwen2-7B-Instruct-AWQ",
    temperature=0
)


runnable = prompt | llm.with_structured_output(schema=SCHEMA_OUTPUT)

for _ in range(5):
    t0 = time.time()
    text = """"🔔QMS NOTIFY
            REPORT: PCV - VISION - Tuần 36/2024
            SCORE: 98.68
            NC: 1
            MESSAGE: PCV đã được PM xác nhận
            ACTION_USER: sonhh20
            DETAIL: https://qms.ghtk.vn/pcv/pcv-report/1488."""
    print(runnable.invoke({"text": text, "examples": []}))
    print("time::: ", time.time() - t0)