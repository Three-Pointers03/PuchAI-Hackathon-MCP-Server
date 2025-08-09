import asyncio
import json
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
print(TOKEN, MY_NUMBER) 
assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert PERPLEXITY_API_KEY is not None, "Please set PERPLEXITY_API_KEY in your .env file"
# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=60,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Analyze Ingredients MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
# JobFinderDescription = RichToolDescription(
#     description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
#     use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
#     side_effects="Returns insights, fetched job descriptions, or relevant job links.",
# )

# @mcp.tool(description=JobFinderDescription.model_dump_json())
# async def job_finder(
#     user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
#     job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
#     job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
#     raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
# ) -> str:
#     """
#     Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
#     """
#     if job_description:
#         return (
#             f"📝 **Job Description Analysis**\n\n"
#             f"---\n{job_description.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**\n\n"
#             f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
#         )

#     if job_url:
#         content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
#         return (
#             f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
#             f"---\n{content.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**"
#         )

#     if "look for" in user_goal.lower() or "find" in user_goal.lower():
#         links = await Fetch.google_search_links(user_goal)
#         return (
#             f"🔍 **Search Results for**: _{user_goal}_\n\n" +
#             "\n".join(f"- {link}" for link in links)
#         )

#     raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

# MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
#     description="Convert an image to black and white and save it.",
#     use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
#     side_effects="The image will be processed and saved in a black and white format.",
# )

# @mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
# async def make_img_black_and_white(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
# ) -> list[TextContent | ImageContent]:
#     import base64
#     import io

#     from PIL import Image

#     try:
#         image_bytes = base64.b64decode(puch_image_data)
#         image = Image.open(io.BytesIO(image_bytes))

#         bw_image = image.convert("L")

#         buf = io.BytesIO()
#         bw_image.save(buf, format="PNG")
#         bw_bytes = buf.getvalue()
#         bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

#         return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

AnalyzeIngredientsDescription = RichToolDescription(
    description="Analyze a product label image for ingredient safety categories using Perplexity (sonar-pro). Returns JSON plus citations.",
    use_when="Use when given a product label image and the user wants a structured ingredient safety analysis.",
    side_effects="Optionally stores to Firestore if configured.",
)

@mcp.tool(description=AnalyzeIngredientsDescription.model_dump_json())
async def analyze_ingredients(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data. Either raw base64 or full data URI starting with 'data:'.")],
    mime_type: Annotated[str, Field(description="MIME type of the image if raw base64 is provided.")] = "image/jpeg",
    user_id: Annotated[str | None, Field(description="User id to associate with the scan if storing.")] = None,
    store: Annotated[bool, Field(description="Store scan to Firestore if configured.")] = True,
) -> str:
    # Accept both raw base64 and data URI
    image_data_uri = puch_image_data if puch_image_data.startswith("data:") else f"data:{mime_type};base64,{puch_image_data}"

    prompt = """
You are an AI assistant that analyzes food product ingredient labels for health and safety. Given a list of ingredients, categorize them into the following:

- Safe
- Low Risk
- Not Great
- Dangerous

Your response MUST be a valid JSON object and contain no additional formatting or markdown. Return ONLY the JSON object. Do not include any text outside the JSON.

Your task includes:

1. Identify the product name from the label (if available).
2. Provide a general safety score (e.g., "Safe: 95%").
3. Provide a 2-3 sentence summary of the ingredient safety profile.
4. For each category (safe, low_risk, not_great, dangerous):
   - Include only the names of ingredients (for use in collapsed UI cards).
   - Include a breakdown with:
     - The ingredient name
     - A short reason for the classification
     - The amount present in the product (if known; otherwise use "unknown")

Additionally:

5. Include an "allergen_additive_warnings" field:
   - A list of any potential allergens (e.g., milk, soy, gluten) or additives (e.g., colorants, preservatives) if mentioned or implied.
   - If none are found, use ["None"].

6. Include a "product_summary" field:
   - A single-sentence summary that briefly describes the nature and safety of the product.

Use this exact JSON structure:

{
  "product_name": "string - Name of the product (if visible on label)",
  "safety_score": "string - e.g. 'Safe: 95%'",
  "ingredients_summary": "string - Overall summary paragraph about safety and risks",
  "ingredient_categories": {
    "safe": {
      "ingredients": ["array of safe ingredient names"],
      "details": [
        {
          "ingredient": "ingredient name",
          "reason": "short explanation of why it's considered safe",
          "amount": "string - amount if known, e.g., '5g' or '2%', else 'unknown'"
        }
      ]
    },
    "low_risk": {
      "ingredients": ["array of low risk ingredient names"],
      "details": [
        {
          "ingredient": "ingredient name",
          "reason": "short explanation of why it's considered low risk",
          "amount": "string - amount if known, e.g., '5g' or '2%', else 'unknown'"
        }
      ]
    },
    "not_great": {
      "ingredients": ["array of not great ingredient names"],
      "details": [
        {
          "ingredient": "ingredient name",
          "reason": "short explanation of why it's considered not great",
          "amount": "string - amount if known, e.g., '5g' or '2%', else 'unknown'"
        }
      ]
    },
    "dangerous": {
      "ingredients": ["array of dangerous ingredient names or ['None'] if none"],
      "details": [
        {
          "ingredient": "ingredient name",
          "reason": "short explanation of why it's considered dangerous",
          "amount": "string - amount if known, e.g., '5g' or '2%', else 'unknown'"
        }
      ]
    }
  },
  "allergen_additive_warnings": ["list of allergens or additives, or ['None']"],
  "product_summary": "string - One sentence describing the product's general purpose and safety"
}
"""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
        "web_search_options": {"search_context_size": "medium"}
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "accept": "application/json",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            data = ""
            resp = await client.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
            # for line in resp.iter_lines():
            #     if line:
            #         line = line.decode('utf-8')
            #         if line.startswith('data: '):
            #             data_str = line[6:]  # Remove 'data: ' prefix
            #             if data_str == '[DONE]':
            #                 break
            #             try:
            #                 chunk_data = json.loads(data_str)
            #                 content = chunk_data['choices'][0]['delta'].get('content', '')
            #                 if content:
            #                     data += content
            #             except json.JSONDecodeError:
            #                 continue

        if resp.status_code >= 400:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Perplexity error {resp.status_code}: {resp.text[:300]}"))
        
        data = resp.json()
        # print(data)
        # analysis_content = (
        #     data.get("choices", [{}])[0]
        #     .get("message", {})
        #     .get("content", "No analysis available")
        # )
        # result = {
        #     "analysis": analysis_content,
        #     "citations": data.get("citations", []),
        # }

        # if store and DB is not None:
        #     try:
        #         # firestore imported only if DB was initialized successfully
        #         from firebase_admin import firestore  # type: ignore
        #         DB.collection("scans").add({
        #             "user_id": user_id or "unknown",
        #             "analysis_result": data,
        #             "timestamp": firestore.SERVER_TIMESTAMP,
        #         })
        #     except Exception:
        #         # Non-fatal if storage fails
        #         pass

        return json.dumps(data)

    except httpx.HTTPError as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Network error: {e!r}"))
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
    

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
