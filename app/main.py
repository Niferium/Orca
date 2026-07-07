import sys
import threading
import itertools
import time

from ollama import chat


class Main:
    def __init__(self):
        self.main_model = "qwen3-coder:30b"
        self.messages = []

    def chat(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            self.messages.append({"role": "user", "content": user_input})
            new_messages = self.generate(self.messages)
            self.messages.extend(new_messages)

    def generate(
        self,
        messages: list,
        tools: list = None,
        max_tokens: int = 2048,
        verbose: bool = False,
    ) -> list:
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(
            target=self._spinner, args=(stop_spinner,)
        )
        spinner_thread.start()

        try:
            stream = chat(
                model=self.main_model,
                messages=messages,
                tools=tools,
                stream=True,
            )

            in_thinking = True
            in_answer = False
            content = ""
            thinking = ""
            first_chunk = True

            for chunk in stream:
                if first_chunk:
                    stop_spinner.set()
                    spinner_thread.join()
                    # clear the spinner line
                    print("\r" + " " * 20 + "\r", end="", flush=True)
                    first_chunk = False

                if chunk.message.thinking:
                    if not in_thinking:
                        in_thinking = True
                        print("Thinking:\n", end="", flush=True)
                    print(chunk.message.thinking, end="", flush=True)
                    thinking += chunk.message.thinking

                elif chunk.message.content:
                    if not in_answer:
                        in_answer = True
                        print("\n\nAnswer:\n", end="", flush=True)
                    print(chunk.message.content, end="", flush=True)
                    content += chunk.message.content

        finally:
            # in case the stream errors out before any chunk arrives
            if not stop_spinner.is_set():
                stop_spinner.set()
                spinner_thread.join()
                print("\r" + " " * 20 + "\r", end="", flush=True)

        print()

        new_messages = [
            {
                "role": "assistant",
                "thinking": thinking,
                "content": content,
            }
        ]

        return new_messages

    @staticmethod
    def _spinner(stop_event: threading.Event):
        for ch in itertools.cycle(["|", "/", "-", "\\"]):
            if stop_event.is_set():
                break
            print(f"\rLoading {ch}", end="", flush=True)
            time.sleep(0.1)


if __name__ == "__main__":
    Main().chat()