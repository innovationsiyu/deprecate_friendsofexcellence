are_adequate_search_results = """The user is using search engines to retrieve information. Please assist in reviewing whether the search results are adequate for the user's search target.

<search_target>
{search_target}
</search_target>

The JSON data you received comprises the obtained search results from search engines.

Based on the "title" and "key_points" of each item, please analyse whether they have covered all the required information for the search target, thereby making the search results adequate.

Give a True or False judgement. If the search results are adequate, simply output "True". If you think there is still specific information that needs to be added, which means the search results are inadequate, simply output "False".

Generate a new query for further search targeting the specific information to be added if the current search results are inadequate. Otherwise, leave the new query field empty.

Output in accordance with the json_schema to ensure the correct format."""
