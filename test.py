from langchain.llms import HuggingFaceHub
from langchain_aws import ChatBedrock
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
import os
import boto3
from getpass import getpass

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceHub(
    repo_id='tiiuae/falcon-7b-instruct',
    model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
)
llm = ChatBedrock(
    client=boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="ASIA6GBMC4A6NWZP6AVU",
        aws_secret_access_key="/CLlDZx5vlPtzEPopTVfEWJzHd73tL5h6/o3opRA",
        aws_session_token="IQoJb3JpZ2luX2VjEH4aCXVzLWVhc3QtMSJHMEUCIQDFRFoz4CsdG9eubdRDeEoBeOnITHNQIL95szzHTdmqqgIgY5Il9RWLfPlnjV3VFZcyJnj6UQuMpfQOk+kpb3xVUi4qlAMIh///////////ARAAGgw5NzUwNTAwMzkzNTYiDLxM7a+ROaXrBE/F9iroAsSRIniH40MxSQdLKJpQxzuybh9eEKnZz2z5rWay1AESD66LdICsQ8TKuNxmbm6VL6DSgfsjAuU8FTl0dI79d648NMxuFphVZ9L9VtYFtYDc6+x+sHgmNCPA6IMK7AYE1B6St53ABWBlnYgv4pD9SsH4d8xVbjoSCbIRjEnz+HUkPDUV0t6Zd+VMKzrLMMiBpp+ZMLcgjgUaED9urYuBVfxxvod4+pqN49r19Wk8aRKM3phJKvRnq+rH34pSpFmoocQCxnyuNFQxjxrpt7MjMYjZd/zGKP8H2nI/afAtrQ5z02FuZyaWJG8oFZWqpVgXJc2z7IZ/5UgAXAxdnn1VMafWbanz1x1OMOTKhxXl8HwcDxzG9bYpryEFDEGzt647jCLAv54/gvGsWTEZJxirim/VMBJ/i6dWz+vUzGuPna4mnq3EMCJhwOoPvzn0OrrSvX2LZ8fNw/aF/cRIFtM4QhFN1uY93wSlxzDtnpu2BjqmAb4cn9D9pfXZvAUMfGeiJnwC8ZF2immPokNPJ4Ch5cuK8EIPPTwaMwEwbv00jiLPIrq7qEIMso5Wk3a0+j1y7XwAIhhD98+P9YMg0rzwKa9yq4b43I0ISaNtRd3byuNeVKKT1LVpmFN4u+3eArU8FAhDMe36j5cv4roL0b43y3X5nHSGJBa65xa30w2MP6xzisKUgmU9FekSWtI+38yB/vcCBlrNEPA="
    ),
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=dict(temperature=0)
)
# llm.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
query = """
diff --git a/app.py b/app.py
    main()
  File "/usr/local/lib/python3.10/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/click/core.py", line 1055, in main
index 97595c3..167f8b5 100644
--- a/app.py
+++ b/app.py
@@ -9,14 +9,15 @@
 #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #  See the License for the specific language governing permissions and
 #  limitations under the License.
-import json
-import os
-from typing import List
+import json
+import os
+from typing import List

-import click
+import click
 import requests
 from langchain import HuggingFaceHub, LLMChain, PromptTemplate
-from loguru import logger
+
+from loguru import logger
 from langchain_google_genai import ChatGoogleGenerativeAI
 
 
@@ -190,4 +191,4 @@ def main(
 
 if __name__ == "__main__":
     # pylint: disable=no-value-for-parameter
-    main()
\ No newline at end of file
+    main()
"""
template = """Provide a concise summary of the bug found in the code, describing its characteristics,
        location, and potential effects on the overall functionality and performance of the application.
        Present the potential issues and errors first, following by the most important findings, in your summary
        Important: Include block of code / diff in the summary also the line number.

        Diff:

        {query}
        """
prompt = ChatPromptTemplate.from_messages(
    [
        
        (
            "human",
            """Provide a concise summary of the bug found in the code, describing its characteristics,
        location, and potential effects on the overall functionality and performance of the application.
        Present the potential issues and errors first, following by the most important findings, in your summary
        Important: Include block of code / diff in the summary also the line number.

        Diff:

        {query}""",
        )
    ]
)
chain = prompt | llm
print(chain.invoke({"query": query}))
# prompt = PromptTemplate(template=template, input_variables=["query"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# review_result = llm_chain.run(query)
print(review_result)
