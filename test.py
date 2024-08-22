from langchain.llms import HuggingFaceHub
from langchain_aws import ChatBedrock
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
import os
import boto3
import requests
import json
from getpass import getpass

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def call_llm_get_results():
    llm = HuggingFaceHub(
        repo_id='tiiuae/falcon-7b-instruct',
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
    )
    llm = ChatBedrock(
        client=boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
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
            the summary should be a json formatted list with objects containing line number and comment for the line
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
    the summary should be a json formatted list with objects containing line number, comment and file name for the line
            Diff:
    
            {query}""",
            )
        ]
    )
    chain = prompt | llm
    result=chain.invoke({"query": query}).content
    print(json.loads(result)[0]["fileName"])
    print(json.loads(result)[0]["lineNumber"])
    print(json.loads(result)[0]["comment"])
    return json.loads(result)[0]["lineNumber"]
    # prompt = PromptTemplate(template=template, input_variables=["query"])
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    # review_result = llm_chain.run(query)
    # print(review_result)


def create_a_comment_to_pull_request(
        github_token: str,
        github_repository: str,
        pull_request_number: int,
        git_commit_hash: str,
        body: str,
        start_line:int,
        path:str):
    """Create a comment to a pull request"""
    headers = {
        "Accept": "application/vnd.github.v3.patch",
        "authorization": f"Bearer {github_token}"
    }
    data = {
        "body": body,
        "commit_id": git_commit_hash,
        
        "line":17,
        "path":path
    }
    print("git commit hash:" + git_commit_hash)
    url = f"https://api.github.com/repos/digitalneedstech/{github_repository}/pulls/{pull_request_number}/comments"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("response:" + response.text)
    return response


if __name__ == '__main__':
    #line =call_llm_get_results()
    
    create_a_comment_to_pull_request(
        github_token="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",
        github_repository="code-review-action",
        pull_request_number=1,
        git_commit_hash="5b45851b8887e374d015f41ed0d6438cb83a800d",
        start_line=17,
        path="app.py",
        body="Missing newline at end of file. This can cause issues with some text editors and version control systems.")
    
