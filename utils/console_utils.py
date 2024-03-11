from rich.console import Console
console = Console()
def print_error(error_message):
    msg = "[ERROR] "
    console.print(f"{msg+error_message}",style="white on red")
def print_success(success_message):
    msg = "[SUCCESS] "
    console.print(f"{msg+success_message}",style="white on green")
def print_warning(warning_message):
    msg = "[WARNING] "
    console.print(f"{msg+warning_message}",style="white on yellow")
def print_info(info_message):
    msg = "[INFO] "
    console.print(f"{msg+info_message}",style="white on blue")
def print_debug(debug_message):
    msg = "[DEBUG] "
    console.print(f"{msg+debug_message}",style="white on purple")