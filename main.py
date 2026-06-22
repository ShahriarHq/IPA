import asyncio
import configparser
import glob
import shutil
import tempfile
from pathlib import Path
from langchain_openai import ChatOpenAI
from browser_use import Agent, ChatOpenAI

# ── Load app config ──────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("config.ini")

API_KEY  = config["IPA"]["api_key"]
MODEL    = config["IPA"]["google_free_model"]
BASE_URL = config["IPA"]["base_url"]
TEMP     = float(config["IPA"]["temperature"])


# ── Load task from command file ───────────────────────────────────
def load_task(command_file: str = "command.ini") -> tuple[str, dict]:
    """
    Reads command.ini and builds a detailed, structured task string.
    Returns a tuple: (task_prompt_string, output_config_dict)
    """
    cmd = configparser.ConfigParser()
    cmd.read(command_file)

    # ── Task Info ────────────────────────────────────────────────
    name        = cmd["TASK_INFO"]["name"]
    description = cmd["TASK_INFO"]["description"]

    # ── Target ───────────────────────────────────────────────────
    url     = cmd["TARGET"]["url"]
    section = cmd["TARGET"]["section"]

    # ── Steps (must be numbered step_1, step_2 ...) ──────────────
    steps_section = dict(cmd["STEPS"])
    steps = []
    i = 1
    while f"step_{i}" in steps_section:
        steps.append(steps_section[f"step_{i}"])
        i += 1

    # ── Output ───────────────────────────────────────────────────
    file_name      = cmd["OUTPUT"]["file_name"]
    save_location  = cmd["OUTPUT"]["save_location"]
    file_format    = cmd["OUTPUT"]["file_format"]
    entry_template = cmd["OUTPUT"]["entry_template"]
    separator      = cmd["OUTPUT"]["separator"]

    # Create the output directory if it does not exist
    output_dir = Path(save_location).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output_path = output_dir / file_name

    # ── Constraints ──────────────────────────────────────────────
    constraints = dict(cmd["CONSTRAINTS"])

    # ── Build the full task prompt ────────────────────────────────
    task  = f"=== TASK: {name} ===\n"
    task += f"GOAL: {description}\n\n"

    task += f"TARGET WEBSITE : {url}\n"
    task += f"TARGET SECTION : {section}\n\n"

    task += "STEPS TO FOLLOW (execute in order, do not skip any step):\n"
    for idx, step in enumerate(steps, 1):
        task += f"  {idx}. {step}\n"

    task += f"\nOUTPUT REQUIREMENTS:\n"
    task += f"  - Save results to file : {file_name}\n"
    task += f"  - Full save path       : {full_output_path}\n"
    task += f"  - File format          : {file_format}\n"
    task += f"  - Each entry format    : {entry_template}\n"
    task += f"  - Between entries use  : {separator}\n"

    task += f"\nCONSTRAINTS (must be respected at all times):\n"
    for key, value in constraints.items():
        task += f"  - {key.replace('_', ' ').title()}: {value}\n"

    # Return both the prompt and the output config for post-run copying
    output_cfg = {
        "file_name":        file_name,
        "full_output_path": full_output_path,
        "output_dir":       output_dir,
    }
    return task.strip(), output_cfg


# ── Post-run: find and copy the output file ───────────────────────
def save_output_file(output_cfg: dict) -> None:
    """
    The browser_use agent saves files to a temp directory.
    This function searches for the file by name and copies it
    to the location defined in command.ini.
    """
    file_name        = output_cfg["file_name"]
    full_output_path = output_cfg["full_output_path"]

    # Gather candidate files from temp dir and current working dir
    temp_base  = tempfile.gettempdir()
    candidates = glob.glob(f"{temp_base}/**/{file_name}", recursive=True)
    candidates += glob.glob(f"**/{file_name}", recursive=True)

    # Exclude the destination itself from candidates
    candidates = [
        p for p in candidates
        if Path(p).resolve() != Path(full_output_path).resolve()
    ]

    if not candidates:
        print(f"⚠️  Could not find '{file_name}' to copy. The agent may not have saved it.")
        return

    # Pick the most recently modified file (freshest agent output)
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    source = candidates[0]

    shutil.copy2(source, full_output_path)
    print(f"\n✅ Output file saved to: {full_output_path}")
    print(f"   (Copied from: {source})")


# ── Main ─────────────────────────────────────────────────────────
async def main():
    task, output_cfg = load_task("command.ini")

    print("=" * 60)
    print("Loaded task from command.ini:\n")
    print(task)
    print("=" * 60)

    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=TEMP,
    )
    agent = Agent(
        task=task,
        llm=llm,
    )
    result = await agent.run()
    print(result)

    # Copy the agent's output file to the desired location
    save_output_file(output_cfg)


asyncio.run(main())