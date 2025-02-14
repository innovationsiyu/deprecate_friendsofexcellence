def reply_with_tool_results2(in_chinese):
    return (
        """你已经执行了工具调用并获得了JSON格式的结果，其中每个键是所执行的函数，对应的值是其返回值。

请回顾聊天历史，专注最近的问题或任务，思考这些信息和数据是否支持你给出符合要求的答案或解决方案。

如果信息和数据已经足够，请以纯文本形式回复。否则，告知工具调用结果不足以支持预期的回复。"""
        if in_chinese else
        """You have made the tool calls and obtained the results in JSON format, in which each key is the function executed and the corresponding value is its return.

Please review the chat history, focus on the preceding question or task, and consider whether the information and data support you in providing an answer or solution.

If there is adequate information and data, reply in plain text. Otherwise, acknowledge that the results are insufficient for an expected reply."""
    )
