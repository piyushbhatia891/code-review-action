from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate,LLMChain
import os
from getpass import getpass

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceHub(
    repo_id='tiiuae/falcon-7b-instruct',
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)
llm.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
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

prompt = PromptTemplate(template=template, input_variables=["query"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
review_result = llm_chain.run(query)
print(review_result)