are_adequate_search_results_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "are_adequate_search_results",
        "schema": {
            "type": "object",
            "properties": {
                "judgement": {
                    "type": "boolean",
                    "description": "A True or False judgement on whether the search results are adequate for the search target."
                },
                "new_query": {
                    "type": "string",
                    "description": "A new query for further search targeting the missing information if the current search results are inadequate. Leave empty if they are adequate."
                }
            },
            "required": ["judgement", "new_query"],
            "additionalProperties": False
        },
        "strict": True
    }
}
