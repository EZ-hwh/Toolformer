import re
from abc import ABCMeta, abstractmethod


class Tool(metaclass=ABCMeta):
    """
    This is the base class for tools. It provides some convenience methods around extracting tool annotations.
    """
    API_CALL_PREFIX = 'Ġ['  # 由于' [' 和 '['表示的是不同的字符，同时在vocab中也是不同的字符，所以这里用'Ġ['来表示' ['，这样就可以保证在tokenizer中不会被分开
    API_CALL_SUFFIX = ']'
    RESULT_PREFIX = '->'

    def get_tool_regex(self, match_before=False) -> str:
        result = r'\[{}\(.*\)\]'.format(self.get_tool_name().upper())
        if match_before:
            result = r'^.*' + result
        return result

    def text_has_call(self, text) -> bool:
        return re.search(self.get_tool_regex(), text) is not None

    def get_call_from_text(self, text) -> str:
        # print(r'^[\s\S]*(?P<api_call>{})'.format(self.get_tool_regex()))
        result = re.search(r'^[\s\S]*(?P<api_call>{})'.format(self.get_tool_regex()), text)
        # print(result)
        return result.groupdict()['api_call']

    def get_text_before_call(self, text) -> str:
        # TODO: refactor
        result = re.search(r'^[\s\S]*(?P<api_call>{})'.format(self.get_tool_regex()), text)
        return text[:result.span('api_call')[0]]

    def get_text_after_call(self, text) -> str:
        result = re.search(r'^[\s\S]*(?P<api_call>{})'.format(self.get_tool_regex()), text)
        return text[result.span('api_call')[1]:]

    @ abstractmethod
    def get_tool_name(self):
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        pass

    @abstractmethod
    def run(self, input: str) -> str:
        pass