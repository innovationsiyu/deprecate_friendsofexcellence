is_matched_search_result_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "is_matched_search_result",
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Your analysis process with the reasons for whether the webpage matches the search target or not."
                },
                "judgement": {
                    "type": "boolean",
                    "description": "A True or False judgement on whether the webpage matches the search target."
                },
                "key_points": {
                    "type": "string",
                    "description": "Multiple key points with elaborations from the webpage if matched. Leave empty if not matched."
                }
            },
            "required": ["analysis", "judgement", "key_points"],
            "additionalProperties": False
        },
        "strict": True
    }
}
