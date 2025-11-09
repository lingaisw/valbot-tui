from pydantic_ai import RunContext
from rich.console import Console
from rich.prompt import Prompt

async def ask_human(
    ctx: RunContext,
    question: str,
    options: list[str],
    default_option: str = None,
    allow_multiple: bool = False,
) -> list[str]:
    """
    Ask the user a question with set choices and get their response.

    Args:
        question (str): The question to ask the user.
        options (list[str]): The options for the user to choose from.
        default_option (str, optional): The default option if the user doesn't provide one. Defaults to None.
        allow_multiple (bool, optional): Whether to allow multiple selections. Defaults to False.

    Returns:
        list[str]: The user's response.
    """
    console = Console()
    console.print(question)
    console.print("Options:")
    for i, option in enumerate(options):
        console.print(f"{i + 1}. {option}")

    if default_option:
        try:
            default_index = options.index(default_option) + 1
            console.print(f"Default option: {default_option} (#{default_index})")
        except ValueError:
            default_index = None
            console.print(f"Default option: {default_option} (not in option list)")
    else:
        default_index = None

    def valid_single_choice(s):
        return s.isdigit() and 1 <= int(s) <= len(options)

    def valid_multiple_choices(s):
        items = [item.strip() for item in s.split(",")]
        return all(item.isdigit() and 1 <= int(item) <= len(options) for item in items if item)

    if allow_multiple:
        while True:
            prompt_str = "Select options by number (comma-separated)"
            user_input = Prompt.ask(prompt_str, default=str(default_index) if default_index else None)
            if valid_multiple_choices(user_input):
                return [options[int(i.strip()) - 1] for i in user_input.split(",") if i.strip()]
            else:
                console.print("[red]Invalid input. Please enter valid option numbers (e.g., 1,2,3).[/red]")
    else:
        while True:
            prompt_str = "Select an option by number"
            user_input = Prompt.ask(prompt_str, default=str(default_index) if default_index else None)
            if valid_single_choice(user_input):
                return [options[int(user_input) - 1]]
            else:
                console.print(f"[red]Invalid input. Please enter a valid option number between 1 and {len(options)}.[/red]")

async def ask_human_for_more_instructions(
    ctx: RunContext,
    question: str,
    default_response: str = None
) -> str:
    """
    Ask the user for more instructions.
    Args:
        question (str): The question to ask the user.
        default_response (str, optional): The default response if the user doesn't provide one. Defaults to None.
    Returns:
        str: The user's response.
    """
    console = Console()
    console.print(question)
    return "The user has reponeded to your question with: " + Prompt.ask("Your response", default=default_response)
