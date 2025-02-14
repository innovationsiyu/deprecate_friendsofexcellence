are_relevant_search_results = """The user is using search engines to retrieve information. Please assist in identifying which of the searched webpages may be relevant to the user's search target:

<search_target>
{search_target}
</search_target>

The JSON data you received comprises some search results from search engines.

Based on the "title" and "summary" of each item, please identify which webpages may be relevant to the search target and warrant a thorough reading for further inspection.

Output the indices of the relevant search results in accordance with the json_schema to ensure the correct format."""
