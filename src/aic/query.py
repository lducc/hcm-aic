from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


class Queries(BaseModel):
    literal: str
    semantic: str
    details: str


def enhance(query):
    load_dotenv()
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=query,
        config={
            "system_instruction": (
                "Rewrite the query for image retrieval. Return three English queries. "
                "literal is a faithful translation. semantic is short and natural. details "
                "emphasizes visible people, objects, actions, colors, counts, locations, "
                "and scene context. Preserve stated facts and never invent details.\n\n"
                "Example input: người phụ nữ cầm ô đỏ dưới mưa\n"
                "literal: a woman holding a red umbrella in the rain\n"
                "semantic: woman with a red umbrella in rain\n"
                "details: a woman outdoors in rainfall, holding a bright red umbrella\n\n"
                "Example input: góc quay từ trên cao của cuộc đua xe đạp\n"
                "literal: an overhead view of a bicycle race\n"
                "semantic: aerial bicycle race\n"
                "details: drone view above cyclists racing together on a road"
                "\n\nExample input: ba tay đua cùng đội áo trắng quần vàng xanh, nón trắng đỏ đen\n"
                "literal: three cyclists from the same team wearing white jerseys and yellow blue shorts, "
                "with white, red, and black helmets\n"
                "semantic: three cyclists in a line from the same team\n"
                "details: overhead view of three cyclists riding in a straight line, white jerseys, "
                "yellow and blue shorts, first rider in a white helmet, second in a red helmet, third in a black helmet"
            ),
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_json_schema": Queries.model_json_schema(),
        },
    )
    result = Queries.model_validate_json(response.text)
    return [result.literal, result.semantic, result.details]
