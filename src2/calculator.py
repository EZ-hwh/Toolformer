from tool import Tool
import re
PROMPT_TEMPLATE = \
"""Your task is to add calls to a Calculator API to a piece of text.
The calls should help you get information required to complete the text.
You can call the API by writing "[CALCULATOR(expression)]" where "expression" is the expression to be computed.
Here are some examples of API calls: 
Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [CALCULATOR(18 + 12 * 3)] 54.
Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
Output: The population is 658,893 people. This is 11.4% of the national average of [CALCULATOR(658,893 / 11.4%)] 5,763,868 people.
Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [CALCULATOR(723 / 252)] 2.87 per match). This is twenty goals more than the [CALCULATOR(723 - 20)] 703 goals last year.
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [CALCULATOR(2011 - 1994)] 17 years.
Input: From this, we have 4 * 30 minutes = 120 minutes.
Output: From this, we have 4 * 30 minutes = [CALCULATOR(4 * 30)] 120 minutes.
Input: {}
Output: """


class CalculatorTool(Tool):
    def get_tool_name(self):
        return 'CALCULATOR'

    def get_prompt_template(self) -> str:
        return PROMPT_TEMPLATE

    def run(self, input: str) -> str:
        # TODO: the following code should be a method in the Tool class.
        # print(input)
        match = re.search(r'^[ ]*CALCULATOR\((.+?)\)[ ]*$', input)
        if match:
            try:
                result = match.group(1)
                return format(eval(result), '.2f')
            except:
                return ''
        else:
            return ''


if __name__ == '__main__':
    print(CalculatorTool().run('CALCULATOR(723010101 / 252000) '))