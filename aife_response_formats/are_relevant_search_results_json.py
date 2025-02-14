are_relevant_search_results_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "are_relevant_search_results",
        "schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "description": "A list of indices for relevant search results. Example: [0, 1, 3, 6]. If no relevant search results are found, output an empty list: [].",
                    "items": {
                        "type": "integer",
                        "description": "Index of a relevant search result. Omit if there are none."
                    }
                }
            },
            "required": ["indices"],
            "additionalProperties": False
        },
        "strict": True
    }
}
