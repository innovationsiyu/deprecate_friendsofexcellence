is_matched_search_result = """The user is using search engines to retrieve information. Please assist in analysing whether each searched webpage matches the user's search target:

<search_target>
{search_target}
</search_target>

What you received is the content scraped from a webpage.

Please review the full content and analyse whether the webpage matches the search target.

Provide your analysis, including the reasons for whether they match or not, before giving a True or False judgement.

If matched, simply output "True". If the content of the webpage falls outside the scope of the search target or merely consists of fragmented text, simply output "False".

If matched, extract multiple key points from the webpage and elaborate on each one. The selection of key points may comprehensively cover the full content or specifically focus on the search target. If not matched, leave the key points field empty.

Output in accordance with the json_schema to ensure the correct format."""
