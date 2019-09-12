import os
import time
import random
import concurrent
import asyncio


def run_parallel(commands, n_worker=1):
    assert commands and type(commands) == list

    def func():
        while True:
            try:
                time.sleep(random.random() * n_worker)
                if not commands:
                    break

                command = commands.pop(random.randrange(len(commands)))
                result_code = os.system(command)
                if result_code != 0:
                    _failed_commands.append(command)

            except KeyboardInterrupt:
                break

    async def main():
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_worker) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(executor, func) for _ in range(n_worker)
            ]

            for response in await asyncio.gather(*futures):
                pass

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


if __name__ == '__main__':
    run_parallel(['ls'] * 100, n_worker=2)
