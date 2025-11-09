
from agent_plugins.common_agent_imports import *
from agent_plugins.agent_plugin import AgentPlugin
import asyncio
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm
from rich import box
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
import readchar
import sys
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from agent_plugins.common_tools.terminal_tools import list_files, find_files, grep_in_files, show_file_tree, run_shell_command
from agent_plugins.common_tools.file_tools import read_file, FileContent

class MyCustomPlugin(AgentPlugin):
    """
    SPF to PDL Converter Agent (Two-Stage Approach)
    
    This agent converts Structural Pattern Framework (SPF) test sequences to 
    Tessent's Procedural Description Language (PDL) format using a two-stage approach:
    
    STAGE 1 - UNDERSTAND: 
        Analyze the SPF code to extract the test intent, objectives, register/signal 
        operations, and sequence logic. This creates a semantic understanding of what 
        the test is trying to accomplish rather than doing direct syntax translation.
    
    STAGE 2 - GENERATE: 
        Using the understanding from Stage 1, generate proper PDL code that accomplishes 
        the same test objectives using correct PDL constructs and idioms.
    
    This approach avoids inconsistent direct translation and ensures the generated 
    PDL uses appropriate commands and syntax.
    """
    
    REQUIRED_ARGS = {}
    
    def get_initializer_args(self):
        """Override to conditionally prompt for input_path only for mode 1."""
        # Import here to avoid issues if not available
        from rich.console import Console
        
        console = Console()
        
        # Menu choices
        choices = [
            ("1", "Convert SPF to PDL"),
            ("2", "PDL Expert Consultation (interactive Q&A)")
        ]
        
        selected_index = 0
        
        def render():
            """Render the menu with current selection highlighted."""
            lines = []
            for i, (key, description) in enumerate(choices):
                if i == selected_index:
                    # Highlight selected option with arrow
                    lines.append(f"  [bold green]→[/bold green]   [bold green]{description}[/bold green]")
                else:
                    # Non-selected option
                    lines.append(f"      [dim]{description}[/dim]")
            return "\n".join(lines)
        
        # Display header
        console.print("\n[bold cyan]Select an option (use ↑ ↓ and Enter)[/bold cyan]\n")
        
        # Interactive selection with arrow keys
        try:
            with Live(render(), refresh_per_second=10, console=console) as live:
                while True:
                    key = readchar.readkey()
                    if key == readchar.key.UP:
                        selected_index = (selected_index - 1) % len(choices)
                    elif key == readchar.key.DOWN:
                        selected_index = (selected_index + 1) % len(choices)
                    elif key == readchar.key.ENTER or key == '\r' or key == '\n':
                        break
                    live.update(render())
        except Exception as e:
            # Fallback to simple input if readchar doesn't work
            console.print(f"[yellow]Arrow key navigation not available: {e}[/yellow]")
            console.print("\n[bold cyan]SPF to PDL Converter Agent[/bold cyan]")
            console.print("[dim]Select a mode to continue[/dim]\n")
            
            table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
            table.add_column("Key", style="cyan", width=5)
            table.add_column("Mode", style="white")
            for key, description in choices:
                table.add_row(key, description)
            
            console.print(table)
            console.print()
            
            mode_value = None
            while mode_value not in ['1', '2']:
                mode_value = console.input("[bold yellow]Select an option (1 or 2):[/bold yellow] ").strip()
                if mode_value not in ['1', '2']:
                    console.print("[red]Invalid selection. Please enter 1 or 2.[/red]")
                    Prompt.ask("retry?")

            
            selected_index = int(mode_value) - 1
        
        mode_value = choices[selected_index][0]
        console.print(f"\n[green]✓ Selected:[/green] {choices[selected_index][1]}\n")
        
        # Only ask for input_path if mode is 1
        if mode_value == '1':
            console.print()
            input_path = console.input("[bold yellow]Enter the path to an SPF file or directory:[/bold yellow] ").strip()
            return {
                'mode': mode_value,
                'input_path': input_path
            }
        else:
            return {'mode': mode_value}

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Initialize Rich console for beautiful formatting
        self.console = Console()
        
        # Create specialized agents for two-stage conversion
        self.understanding_agent = self.create_understanding_agent(model)
        self.pdl_generation_agent = self.create_pdl_generation_agent(model)
        # Legacy: keep for backward compatibility
        self.converter_agent = self.create_agent(model)
        # PDL Expert for interactive consultation
        self.pdl_expert_agent = self.create_pdl_expert_agent(model)
        # Provide tools for agent to use autonomously
        self.add_agentic_tools()

    def create_agent(self, model):
        """Create the SPF to PDL conversion agent with specialized system prompt."""
        system_prompt = """
        You are an expert in pre-silicon validation and test pattern languages.
        Your task is to understand SPF test sequences and generate equivalent PDL code.
        
        IMPORTANT: Do NOT attempt direct translation. Instead:
        1. First understand what the SPF is trying to accomplish
        2. Then write PDL using proper PDL constructs for that purpose
        
        You will work in TWO stages with relevant Tessent manual knowledge provided.
        """
        
        return Agent(
            model=model,
            system_prompt=system_prompt,
            retries=3
        )
    
    def create_understanding_agent(self, model):
        """Create an agent specialized in understanding SPF intent."""
        system_prompt = """
        You are an expert in analyzing Structural Pattern Framework (SPF) test sequences.
        
        Your task is to analyze SPF code and extract ONLY the test operations and data.
        Do NOT translate to PDL. Do NOT mention any PDL commands or syntax.
        
        Focus ONLY on:
        - Which registers/signals are accessed (names and addresses)
        - What values are written (exact hex/binary values)
        - What values are read/checked (expected values)
        - What timing delays are needed (exact durations)
        - The sequence of operations (step by step)
        - Any loops or conditional logic
        - SPF label values (these will be used for documentation/comments)
        
        CRITICAL: Distinguish between register fields and bit numbers:
        - Register FIELD: Named field in register (e.g., register.FIELD_NAME or register.Enable)
        - Bit NUMBER: Numeric bit position (e.g., register[7] or register[15:8])
        - Do NOT confuse bit numbers with field names
        - Preserve the exact notation used in SPF (brackets for bits, dots for fields)
        
        CRITICAL: Extract exact value formats with proper prefixes:
        - Binary values: MUST have 0b prefix (e.g., 0b1010, 0b0, 0b111)
        - Hexadecimal values: MUST have 0x prefix (e.g., 0x1234, 0xABCD, 0x0)
        - Decimal values: Use 0d prefix explicitly (e.g., 0d10, 0d255) or no prefix for simple decimals
        - Preserve the exact prefix format from SPF
        
        OUTPUT FORMAT:
        === TEST OBJECTIVE ===
        [High-level purpose in one sentence]
        
        === OPERATIONS ===
        [List each operation as: WRITE register.field = value, READ register.field expect value, WAIT time, CALL procedure, etc.]
        [For bit access use: WRITE register[bit] or register[high:low], not register.bit]
        [Include LABEL: "text" for each operation that has an SPF label]
        
        === SEQUENCE ===
        1. [First operation with data] LABEL: "spf_label_text"
        2. [Second operation with data] LABEL: "spf_label_text"
        ...
        
        === TIMING ===
        [Any timing constraints or delays]
        
        CRITICAL RULES:
        1. Focus on WHAT data goes WHERE, not on how to implement it
        2. Extract and preserve any SPF label values - they describe what each step does
        3. PRESERVE EXACT signal paths and register names as they appear in SPF - do NOT modify, shorten, or rename them
        4. Keep the full hierarchical paths exactly as written (e.g., module.submodule.register.field)
        5. DO NOT convert bit numbers to field names - keep [bit] notation for bit access
        6. Use register.FIELD_NAME only when SPF uses named fields
        """
        
        return Agent(
            model=model,
            system_prompt=system_prompt,
            retries=2
        )
    
    def create_pdl_generation_agent(self, model):
        """Create an agent specialized in generating PDL from requirements."""
        system_prompt = """
        You are an expert in Tessent Procedural Description Language (PDL).
        
        You will receive:
        1. A description of test operations (register names, values, timing)
        2. Example PDL code and syntax from Tessent manuals
        
        Your task is to write PDL code using ONLY the commands and syntax shown in the examples.
        
        CRITICAL RULES FOR PDL CODE GENERATION:
        1. ONLY use PDL commands that appear in the provided manual excerpts
        2. DO NOT invent or hallucinate PDL functions
        3. Follow the EXACT syntax patterns from the examples
        4. Use the TCL-like syntax shown (iProc, iWrite, iRead, iCall, iApply, etc.)
        5. If a command is not in the examples, do NOT use it
        6. ALWAYS start PDL files with these TWO REQUIRED PDL header lines (in this exact order):
           iProcsForModule [get_single_name [get_current_design -icl]] 
           source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl
           NOTE: These headers are PDL-specific, NOT used in SPF
        7. PRIORITIZE iSim commands for signal polling, peeking, and macro mapping
        8. COMPULSORY: Validate iSim command parameters match examples exactly:
           - iSim poll_signal: signal_path expected_value timeout step_size
           - iSim peek_signal: signal_path expected_value
           - iSim macro_map: alias rtl_path
        9. CRITICAL: Use iSim macro_map for repeating signal paths:
           - Identify long or repeated signal hierarchies in the test operations
           - Create macro mappings at the start of procedures using: iSim macro_map alias full_rtl_path
           - Use the backtick prefix (`) when referencing mapped macros in subsequent commands
           - Example pattern from example.pdl:
             iSim macro_map man_fuse_sip_rel_req pcd_tb.pcd.parfuse.parfuse_pwell_wrapper.fuse_top1.i_chassis_fuse_controller_top.i_fuse_array_cntrl.i_fuse_array_cntrl_tap.man_fuse_sip_release_req
             iSim poll_signal `man_fuse_sip_rel_req 0x1 10us 5ns
           - This improves code readability and maintainability
        10. Use iNote "description" in PDL code to document what each section or operation is doing
        10. When converting from SPF, use the SPF label values as descriptions in PDL iNote commands
            NOTE: iNote is a PDL command for documentation, NOT an SPF command
        11. PRESERVE EXACT signal paths and register names from the test operations - do NOT modify, rename, or shorten them
        12. Keep full hierarchical paths exactly as provided (e.g., pcd_tb.pcd.module.register.field)
        13. CRITICAL: Distinguish between register fields and bit numbers:
            - Use register.FIELD_NAME for named fields (e.g., register_name.Enable)
            - DO NOT treat bit numbers as field names
            - If test operations show [bit] notation, that's a bit number, NOT a field name
            - Do NOT convert register[7] to register.7 or register."7"
        14. CRITICAL: Use explicit prefixes for all PDL values:
            - Binary values: ALWAYS use 0b prefix (0b1010, 0b0, 0b111, 0b11111111)
            - Hexadecimal values: ALWAYS use 0x prefix (0x1234, 0xABCD, 0x0, 0xFFFF)
            - Decimal values: Use 0d prefix explicitly (0d10, 0d100) or no prefix for clarity
            - NEVER use values without proper prefixes (e.g., write 0b1 not just 1, write 0x10 not just 10)
            - Check PDL examples for proper value format patterns
        15. COMPULSORY: Check the provided manual examples for EXACT parameter requirements:
            - What parameters each command requires (order, count, type)
            - What format values must be in (hex 0x..., binary 0b..., decimal, string)
            - What data types are expected (integer, string, list, etc.)
            - Match parameter formats EXACTLY as shown in examples
        
        Common PDL commands you should look for in examples (PRIORITIZE iSim commands):
        - iSim commands (CHECK THESE FIRST)  # Simulation commands (DTEG dftPdl) - PRIORITY
          - iSim poll_signal signal_path expected_value timeout step_size
          - iSim peek_signal signal_path expected_value
          - iSim macro_map alias rtl_path  # Define short aliases for long signal paths
            * Place macro_map at start of procedures (often in initialization iProc like fuse_rtl_map)
            * Reference with backtick: `alias_name
            * Example: iSim macro_map my_signal pcd_tb.pcd.module.submodule.long_signal_name
            * Then use: iSim poll_signal `my_signal 0x1 10us 5ns
        - iProc <name> {args} { ... }  # Define procedure
        - iWrite register.field value   # Write to register field
        - iRead register.field value    # Read and check register field  
        - iApply                        # Apply the register operations
        - iCall procedure args          # Call another procedure
        - iRunLoop count -tck           # Run clock cycles
        - iNote "message"               # Add comment/note in PDL (USE THIS FREQUENTLY IN PDL CODE)
        
        BEFORE writing code, CHECK the examples for:
        - iSim command parameters FIRST (poll_signal, peek_signal, macro_map)
        - Opportunities to use macro_map for repeated/long signal paths
        - Parameter count and order for each command
        - Value formats (0x for hex, 0b for binary, decimal, quotes for strings)
        - Required vs optional parameters
        - Proper syntax for loops, conditionals, variable assignments
        
        OUTPUT FORMAT:
        === PDL CODE ===
        iProcsForModule [get_single_name [get_current_design -icl]] 
        source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl
        
        [Rest of PDL code using ONLY commands from the manual examples]
        [PRIORITIZE iSim commands where applicable]
        [Include iNote "description" statements throughout to document PDL operations]
        [NOTE: iNote is PDL-specific syntax for documentation, not used in SPF]
        [Use EXACT signal paths and register names as provided in test operations]
        [Use EXACT parameter formats as shown in examples]
        [Validate iSim command parameters match examples]
        
        === IMPLEMENTATION NOTES ===
        [List which PDL commands you used and where you found them in the examples]
        [If iSim commands used, confirm their parameters match examples exactly]
        [Confirm parameter formats match the examples (hex/binary/decimal/string)]
        """
        
        return Agent(
            model=model,
            system_prompt=system_prompt,
            retries=3
        )
    
    def create_pdl_expert_agent(self, model):
        """Create an interactive PDL expert agent for consultation."""
        system_prompt = """
        You are an expert consultant on Tessent Procedural Description Language (PDL).
        
        Users will ask you questions about PDL syntax, commands, best practices, or how to 
        implement specific test scenarios. You will be provided with relevant excerpts from 
        Tessent manuals, DTEG ITPP Reader Commands documentation, and example PDL code to answer their questions.
        
        Your role:
        - Answer questions about PDL syntax and commands
        - Explain how PDL constructs work
        - Provide code examples using ONLY commands from the manual excerpts
        - Help debug PDL issues
        - Suggest best practices for test implementation
        - Guide users on proper PDL patterns
        - Explain DTEG iSim commands for simulation control
        
        IMPORTANT RULES:
        1. Base your answers on the provided manual excerpts and examples
        2. Use ONLY PDL commands that exist in the documentation
        3. Cite which manual section or example you're referencing
        4. When referencing files from the knowledge base (e.g., example.pdl), ALWAYS include the full file path for user reference
        5. If you don't have information in the provided context, say so
        6. Provide clear, actionable code examples when relevant
        7. ALWAYS include the required TWO PDL header lines in PDL code examples:
           iProcsForModule [get_single_name [get_current_design -icl]] 
           source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl
           NOTE: These headers are required for PDL files only, not for SPF files
        8. Use iNote "description" liberally to document what PDL code is doing
           NOTE: iNote is a PDL command for documentation, not used in SPF
        
        CRITICAL: Format ALL responses using proper Markdown:
        - Use ### for section headers
        - Use ```pdl for PDL code blocks
        - Use `command` for inline command names
        - Use **bold** for emphasis
        - Use bullet points (- or *) for lists
        - Use > for important notes or warnings
        
        Example response format:
        The `iWrite` command is used to write values to register fields.
        
        ```pdl
        iProcsForModule [get_single_name [get_current_design -icl]] 
        source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl
        
        iNote "Write to register field"
        iWrite register_name.field_name value
        iApply
        ```
        
        **Key points:**
        - Multiple `iWrite` commands can be grouped
        - `iApply` executes the writes
        
        > **Note:** Always call `iApply` after `iWrite` operations
        
        **Reference:** example.pdl  
        **Full Path:** C:/path/to/agents/example.pdl (line 45)
        
        Common PDL commands to reference:
        - iProc, iWrite, iRead, iApply, iCall, iRunLoop, iNote
        - iSim commands (DTEG dftPdl):
          - iSim poll_signal signal_path expected_value timeout step_size
          - iSim peek_signal signal_path expected_value
          - iSim macro_map alias rtl_path
            * Use to create short aliases for long/repeated signal paths
            * Define at start of procedures (e.g., in initialization iProc)
            * Reference with backtick prefix: `alias_name
        - TCL constructs: set, foreach_in_collection, if/else, for
        - get_icl_instance, get_icl_module, get_single_name
        
        Be helpful, accurate, and always ground your answers in the documentation.
        Use proper Markdown formatting in ALL responses.
        """
        
        return Agent(
            model=model,
            system_prompt=system_prompt,
            retries=2
        )
    
    def add_agentic_tools(self):
        """
        Add autonomous tools for the agent to use during conversion.
        These tools help with file parsing and structure analysis.
        """
        # Tools can be added here if needed for autonomous operation
        self.pdl_expert_agent.tool(list_files)
        self.pdl_expert_agent.tool(find_files)
        self.pdl_expert_agent.tool(grep_in_files)
        self.pdl_expert_agent.tool(show_file_tree)
        self.pdl_expert_agent.tool(run_shell_command)
        self.pdl_expert_agent.tool(read_file)

    def find_spf_files(self, input_path: str) -> List[str]:
        """
        Find all SPF files in the given path.
        
        Args:
            input_path: File path or directory path
            
        Returns:
            List of SPF file paths
        """
        spf_files = []
        path = Path(input_path)
        
        if path.is_file():
            if path.suffix.lower() == '.spf':
                spf_files.append(str(path))
            else:
                self.console.print(f"[yellow]Warning: {input_path} is not an SPF file[/yellow]")
        elif path.is_dir():
            self.console.print(f"[cyan]Scanning directory:[/cyan] {input_path}")
            spf_files = [str(f) for f in path.rglob('*.spf')]
            self.console.print(f"[green]Found {len(spf_files)} SPF files[/green]")
        else:
            self.console.print(f"[red]Error: Path does not exist: {input_path}[/red]")
        
        return spf_files

    def read_spf_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read and parse SPF file structure.
        
        Args:
            file_path: Path to SPF file
            
        Returns:
            Dictionary containing file metadata and content
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic structure analysis
            spf_data = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "content": content,
                "line_count": len(content.split('\n')),
                "has_macros": "MACRO" in content.upper(),
                "has_vectors": "VECTOR" in content.upper(),
                "has_scan": "SCAN" in content.upper(),
                "size_bytes": len(content)
            }
            
            return spf_data
        except Exception as e:
            self.console.print(f"[red]Error reading {file_path}: {e}[/red]")
            return None

    def analyze_spf_structure(self, spf_data: Dict) -> str:
        """
        Analyze SPF file structure to provide context for conversion.
        
        Args:
            spf_data: SPF file data dictionary
            
        Returns:
            Analysis summary string
        """
        analysis = f"""
        SPF File Analysis:
        - File: {spf_data['file_name']}
        - Lines: {spf_data['line_count']}
        - Size: {spf_data['size_bytes']} bytes
        - Contains Macros: {spf_data['has_macros']}
        - Contains Vectors: {spf_data['has_vectors']}
        - Contains Scan Operations: {spf_data['has_scan']}
        """
        
        # Extract key patterns
        content = spf_data['content']
        patterns = {
            'CALL': len(re.findall(r'\bCALL\b', content, re.IGNORECASE)),
            'MACRO': len(re.findall(r'\bMACRO\b', content, re.IGNORECASE)),
            'IOPIN': len(re.findall(r'\bIOPIN\b', content, re.IGNORECASE)),
            'TIMING': len(re.findall(r'\bTIMING\b', content, re.IGNORECASE)),
        }
        
        if any(patterns.values()):
            analysis += "\n        Key Commands Found:\n"
            for cmd, count in patterns.items():
                if count > 0:
                    analysis += f"        - {cmd}: {count} occurrences\n"
        
        return analysis

    def convert_spf_to_pdl(self, spf_data: Dict) -> Dict[str, Any]:
        """
        Convert SPF content to PDL using two-stage AI approach:
        1. Understand what the SPF does
        2. Generate PDL based on that understanding
        
        Args:
            spf_data: SPF file data dictionary
            
        Returns:
            Dictionary containing conversion results
        """
        # Analyze structure first
        analysis = self.analyze_spf_structure(spf_data)
        
        try:
            # ====================================================================
            # STAGE 1: UNDERSTAND SPF INTENT
            # ====================================================================
            self.console.print(f"  [cyan]Stage 1:[/cyan] Understanding SPF intent...")
            
            understanding_prompt = f"""
            Analyze the following SPF test code and explain what it does.
            Focus on the test objectives, register/signal operations, and sequence.
            
            {analysis}
            
            SPF CONTENT:
            {spf_data['content']}
            
            Provide a clear description of the test's purpose and operations.
            """
            
            understanding_result = asyncio.run(self.understanding_agent.run(understanding_prompt))
            test_understanding = understanding_result.data
            
            self.console.print(f"  [green]✓[/green] SPF intent analyzed")
            
            # ====================================================================
            # STAGE 2: GENERATE PDL FROM UNDERSTANDING
            # ====================================================================
            self.console.print(f"  [cyan]Stage 2:[/cyan] Generating PDL from understanding...")
            
            # Create a search query based on the operations found
            search_terms = []
            content_lower = spf_data['content'].lower()
            
            # Identify key operations
            if 'scan' in content_lower:
                search_terms.append('scan')
            if 'vector' in content_lower or 'pattern' in content_lower:
                search_terms.append('vector pattern')
            if 'macro' in content_lower or 'call' in content_lower:
                search_terms.append('iProc procedure call iCall')
            if 'iopin' in content_lower or 'signal' in content_lower:
                search_terms.append('iWrite iRead signal register')
            if 'timing' in content_lower or 'clock' in content_lower:
                search_terms.append('iRunLoop timing clock wait')
            if 'register' in content_lower or 'reg' in content_lower:
                search_terms.append('iWrite iRead iApply register')
            
            # Always search for core PDL syntax
            search_terms.append('iProc iWrite iRead iApply iCall iRunLoop iNote')
            
            # Generate PDL based on understanding
            pdl_generation_prompt = f"""
            Write PDL code to implement the following test operations.
            
            TEST OPERATIONS TO IMPLEMENT:
            {test_understanding}
            
            CRITICAL INSTRUCTIONS:
            1. Use standard PDL commands and syntax
            2. Follow PDL patterns (iProc, iWrite, iRead, iCall, iApply, etc.)
            3. Match the TCL-like syntax structure
            4. ALWAYS start with these TWO header lines (in this exact order):
               iProcsForModule [get_single_name [get_current_design -icl]] 
               source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl
            5. Use iSim commands when applicable (poll_signal, peek_signal, macro_map)
            6. iSim command parameters:
               - iSim poll_signal: requires signal_path, expected_value, timeout, step_size
               - iSim peek_signal: requires signal_path, expected_value
               - iSim macro_map: requires alias, rtl_path
            7. Use iSim macro_map for repeating signal paths:
               - If signal hierarchies are long or used multiple times, create macro mappings
               - Place macro_map commands at the start of procedures
               - Pattern: iSim macro_map short_alias full.hierarchical.path.to.signal
               - Reference mapped signals with backtick prefix: `short_alias
            8. Use iNote "description" in your PDL code to document what each section is doing
            9. If converting from SPF, use the SPF label values in your PDL iNote descriptions
                NOTE: iNote is a PDL command - it's not used in SPF source files
            10. PRESERVE EXACT signal paths and register names - do NOT modify, rename, or shorten them
            11. Copy signal/register names character-for-character from the test operations description
            12. Distinguish bit numbers from field names:
                - If test operations show register[7] or register[15:8], that's BIT ACCESS, not a field
                - Use register.FIELD_NAME only for actual named fields (e.g., register.Enable)
                - DO NOT convert bit numbers to field names (register[7] ≠ register.7)
            13. Use explicit prefixes for ALL PDL values:
                - Binary values: ALWAYS use 0b prefix (0b1, 0b0, 0b1010, 0b11111111)
                - Hexadecimal values: ALWAYS use 0x prefix (0x0, 0x1234, 0xABCD, 0xFFFFFFFF)
                - Decimal values: Use 0d prefix or no prefix (0d10, 0d255, or 100)
                - NEVER write ambiguous values - always make the base explicit
            14. Check parameter requirements BEFORE writing each command:
                - How many parameters does each command need?
                - What format should values be in? (0x1234 for hex, 0b1010 for binary, decimal, "string")
                - What is the correct parameter order?
            
            PDL command reference:
            - iProc: Define procedures
            - iWrite: Write to registers
            - iRead: Read registers
            - iApply: Apply operations
            - iRunLoop: Add timing delays
            - iCall: Call procedures
            - iNote: Add descriptive notes
            - iSim poll_signal: Poll signal until expected value
            - iSim peek_signal: Check signal value
            - iSim macro_map: Map signal aliases
            
            Format your response EXACTLY as:
            === PDL CODE ===
            iProcsForModule [get_single_name [get_current_design -icl]] 
            source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl
            
            [Rest of PDL code using standard PDL commands]
            [Use iSim commands where applicable]
            [Use iSim macro_map for any repeated or long signal paths - define mappings early]
            [Reference mapped signals with backtick prefix: `alias_name]
            [Include iNote statements to describe what each PDL section does]
            [NOTE: The two header lines and iNote are PDL-specific, not used in SPF]
            [Use EXACT signal paths and register names from test operations - copy them exactly]
            [Use correct value formats for each parameter]
            [Ensure iSim commands have all required parameters]
            [Use register.FIELD_NAME for named fields, NOT for bit numbers]
            [Use explicit prefixes: 0b for binary, 0x for hex, 0d for decimal]
            
            === IMPLEMENTATION NOTES ===
            [List which PDL commands you used]
            [If iSim commands used, list them and confirm parameters are complete and correct]
            [If iSim macro_map used, list the mappings created and why they were beneficial]
            [Confirm that all signal paths and register names match the test operations exactly]
            [Confirm parameter formats used (hex/binary/decimal/string)]
            [List what parameter formats you used for each command]
            [Confirm bit numbers vs field names are correctly distinguished]
            [Confirm all values use explicit prefixes (0b, 0x, 0d)]
            """
            
            pdl_result = asyncio.run(self.pdl_generation_agent.run(pdl_generation_prompt))
            
            self.console.print(f"  [green]✓[/green] PDL code generated")
            
            # Parse the result
            pdl_code = ""
            implementation_notes = ""
            
            response = pdl_result.data
            if "=== PDL CODE ===" in response:
                parts = response.split("=== PDL CODE ===")
                if len(parts) > 1:
                    rest = parts[1]
                    if "=== IMPLEMENTATION NOTES ===" in rest:
                        code_part, notes_part = rest.split("=== IMPLEMENTATION NOTES ===", 1)
                        pdl_code = code_part.strip()
                        implementation_notes = notes_part.strip()
                    else:
                        pdl_code = rest.strip()
            else:
                # If format not followed, use entire response as code
                pdl_code = response
            
            # Combine both token usage
            total_tokens = understanding_result.usage().total_tokens + pdl_result.usage().total_tokens
            
            return {
                "success": True,
                "spf_file": spf_data['file_path'],
                "test_understanding": test_understanding,
                "pdl_code": pdl_code,
                "implementation_notes": implementation_notes,
                "conversion_notes": f"Two-stage conversion:\n\n{test_understanding}\n\n{implementation_notes}",
                "tokens_used": total_tokens
            }
        except Exception as e:
            self.console.print(f"  [red]Conversion failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "spf_file": spf_data['file_path'],
                "error": str(e)
            }

    def save_pdl_file(self, conversion_result: Dict, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Save converted PDL to file along with understanding and notes.
        
        Args:
            conversion_result: Conversion result dictionary
            output_dir: Optional output directory (defaults to current working directory)
            
        Returns:
            Path to saved PDL file
        """
        if not conversion_result.get('success'):
            return None
        
        spf_path = Path(conversion_result['spf_file'])
        
        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = Path.cwd()  # Use current working directory
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create PDL filename
        pdl_filename = spf_path.stem + '.pdl'
        pdl_path = out_dir / pdl_filename
        
        # Write PDL file
        try:
            pdl_code = conversion_result['pdl_code']
            
            # Ensure the required TWO header lines are at the top
            required_line1 = "iProcsForModule [get_single_name [get_current_design -icl]]"
            required_line2 = "source $::env(DUVE_M_HOME)/verif/pdl/common/tap_utils.pdl"
            
            # Remove any leading whitespace
            pdl_code = pdl_code.strip()
            
            # Check if headers are already present
            has_line1 = pdl_code.startswith(required_line1)
            has_line2 = required_line2 in pdl_code.split('\n')[:5]  # Check first 5 lines
            
            if not (has_line1 and has_line2):
                # Remove headers if partially present
                lines = pdl_code.split('\n')
                filtered_lines = []
                for line in lines:
                    if line.strip() != required_line1 and line.strip() != required_line2:
                        filtered_lines.append(line)
                
                # Add both headers at the top
                pdl_code = required_line1 + "\n" + required_line2 + "\n\n" + '\n'.join(filtered_lines)
            
            with open(pdl_path, 'w', encoding='utf-8') as f:
                f.write(pdl_code)
            
            # Save test understanding if available
            if conversion_result.get('test_understanding'):
                understanding_path = out_dir / (spf_path.stem + '_test_understanding.txt')
                with open(understanding_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"SPF Test Understanding: {spf_path.name}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(conversion_result['test_understanding'])
                    f.write("\n\n" + "=" * 80 + "\n")
                    f.write("PDL Implementation Notes\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(conversion_result.get('implementation_notes', ''))
            
            # Save conversion notes if available (backward compatibility)
            elif conversion_result.get('conversion_notes'):
                notes_path = out_dir / (spf_path.stem + '_conversion_notes.txt')
                with open(notes_path, 'w', encoding='utf-8') as f:
                    f.write(conversion_result['conversion_notes'])
            
            return str(pdl_path)
        except Exception as e:
            self.console.print(f"[red]Error saving PDL file: {e}[/red]")
            return None

    def run_pdl_expert_mode(self, context, **kwargs):
        """
        Interactive PDL expert consultation mode with conversation history.
        Users can ask questions about PDL and get answers.
        The agent remembers previous questions and answers in the session.
        """
        from rich.markdown import Markdown
        
        # Display header
        self.console.rule("[bold blue]PDL Expert Consultation Mode[/bold blue]", style="blue")
        self.console.print("\n[bold cyan]PDL Expert Consultation Mode - Interactive Q&A[/bold cyan]")
        self.console.print("\n[bold yellow]📝 Note:[/bold yellow] This mode works best in CLI. In TUI, the interactive prompt loop is not yet supported.")
        self.console.print("\n[dim]Ask questions about Tessent PDL syntax, commands, and best practices.[/dim]")
        self.console.print("[dim]  • Press alt+Enter to submit your question or Enter for a new line[/dim]")
        self.console.print("[dim]  • Type 'exit', 'quit', or 'done' to end the session[/dim]")
        self.console.print("[dim]  • The agent remembers your conversation history[/dim]\n")
        
        # Interactive loop with conversation history
        session_history = []
        conversation_context = []  # For maintaining conversation memory
        question_count = 0
        
        while True:
            try:
                # Get user question with multiline support
                # Use prompt_toolkit for proper Shift+Enter support
                self.console.print("[bold yellow]Your question:[/bold yellow]")
                user_question = prompt("", multiline=True).strip()
                
                if not user_question:
                    continue
                
                question_count += 1
                
                # Build conversation history for context
                conversation_history_text = ""
                if conversation_context:
                    conversation_history_text = "\n\nPREVIOUS CONVERSATION IN THIS SESSION:\n"
                    for i, conv in enumerate(conversation_context[-3:], 1):  # Last 3 exchanges for context
                        conversation_history_text += f"\nQ{i}: {conv['question']}\nA{i}: {conv['answer'][:300]}...\n"
                
                # Build consultation prompt with conversation history
                consultation_prompt = f"""
                {conversation_history_text}
                
                CURRENT USER QUESTION: {user_question}
                
                Please answer the user's question about PDL (Procedural Description Language).
                If the question refers to previous conversation (e.g., "what about...", "can you explain more...", "how does that work..."),
                use the conversation history context to provide a coherent response.
                
                Provide clear explanations and code examples when appropriate.
                Use standard PDL commands and syntax including:
                - iProc, iWrite, iRead, iApply, iCall, iRunLoop, iNote
                - iSim commands: poll_signal, peek_signal, macro_map
                
                IMPORTANT: 
                - Format your response using proper Markdown syntax
                - Provide practical code examples where relevant
                - Maintain context from previous questions in this session when relevant
                """
                
                # Get answer from PDL expert agent with spinner
                with self.console.status("[bold cyan]Just a sec...", spinner="dots") as status:
                    result = asyncio.run(self.pdl_expert_agent.run(consultation_prompt))
                
                # Display answer with markdown formatting
                self.console.rule("[bold green]Answer[/bold green]", style="green")
                
                # Render as markdown
                md = Markdown(result.data)
                self.console.print(md)
                self.console.print()
                
                # Save to conversation context (for maintaining conversation memory)
                conversation_context.append({
                    "question": user_question,
                    "answer": result.data
                })
                
                # Save to session history (for final summary)
                session_history.append({
                    "question": user_question,
                    "answer": result.data,
                    "tokens": result.usage().total_tokens
                })
                
            except KeyboardInterrupt:
                self.console.print("\n\n[cyan]Session interrupted. Goodbye![/cyan]")
                break
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]\n")
                import traceback
                traceback.print_exc()
        
        # Display session summary
        if session_history:
            self.console.rule("[bold blue]Session Summary[/bold blue]", style="blue")
            total_tokens = sum(item['tokens'] for item in session_history)
            self.console.print(f"""
[bold green]Summary:[/bold green]
  • Questions asked: {question_count}
  • Total tokens used: {total_tokens}
            """)
            
            # Save session history
            try:
                history_file = Path.cwd() / f"pdl_expert_session_{Path.cwd().name}.json"
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(session_history, f, indent=2)
                self.console.print(f"[green]Session history saved to:[/green] [bold]{history_file}[/bold]")
            except Exception as e:
                self.console.print(f"[yellow]Could not save session history: {e}[/yellow]")

    def run_agent_flow(self, context, **kwargs):
        """
        Main entry point - routes to appropriate mode based on user selection.
        """
        # Get mode from initializer_args (populated by get_initializer_args)
        mode = self.initializer_args.get('mode')
        
        # If no mode, call get_initializer_args to prompt user
        if not mode:
            init_args = self.get_initializer_args()
            self.initializer_args.update(init_args)
            mode = self.initializer_args.get('mode', '1')
        
        if mode == '2':
            # PDL Expert consultation mode
            return self.run_pdl_expert_mode(context, **kwargs)
        else:
            # SPF to PDL conversion mode (default)
            return self.run_spf_conversion_mode(context, **kwargs)
    
    def run_spf_conversion_mode(self, context, **kwargs):
        """
        SPF to PDL conversion workflow.
        """
        # Get input_path from initializer_args
        input_path = self.initializer_args.get('input_path')
        
        # Also check kwargs in case it's passed differently
        if not input_path:
            input_path = kwargs.get('input_path')
        
        if not input_path:
            self.console.print("[red]Error: input_path is required for conversion mode[/red]")
            self.console.print(f"[dim]Debug: initializer_args = {self.initializer_args}[/dim]")
            self.console.print(f"[dim]Debug: kwargs = {kwargs}[/dim]")
            return
        
        # Display header
        self.console.rule("[bold blue]SPF to PDL Converter Agent[/bold blue]", style="blue")
        self.console.print(f"\n[bold cyan]Input Path:[/bold cyan] {input_path}\n")
        
        # Stage 1: Find SPF files
        self.console.print("[bold yellow]STAGE 1: Finding SPF Files[/bold yellow]", justify="center")
        spf_files = self.find_spf_files(input_path)
        
        if not spf_files:
            self.console.print("\n[bold red]No SPF files found![/bold red]")
            return
        
        self.console.print(f"\n[bold green]Found {len(spf_files)} SPF file(s) to convert[/bold green]\n")
        
        # Stage 2: Convert each file
        self.console.print("[bold yellow]STAGE 2: Converting SPF to PDL (Two-Stage Approach)[/bold yellow]", justify="center")
        self.console.print("[dim]Stage 2a: Understanding SPF intent | Stage 2b: Generating PDL[/dim]\n", justify="center")
        
        conversion_results = []
        total_tokens = 0
        
        for idx, spf_file in enumerate(spf_files, 1):
            self.console.print(f"\n[bold white]({idx}/{len(spf_files)})[/bold white] Processing: [cyan]{Path(spf_file).name}[/cyan]")
            
            # Read SPF file
            spf_data = self.read_spf_file(spf_file)
            if not spf_data:
                continue
            
            # Convert to PDL
            result = self.convert_spf_to_pdl(spf_data)
            conversion_results.append(result)
            
            if result.get('success'):
                total_tokens += result.get('tokens_used', 0)
                self.console.print(f"  [green]✓ Conversion successful[/green]")
            else:
                self.console.print(f"  [red]✗ Conversion failed: {result.get('error')}[/red]")
        
        # Stage 3: Save results
        self.console.print(f"\n[bold yellow]STAGE 3: Saving PDL Files[/bold yellow]", justify="center")
        
        saved_files = []
        failed_files = []
        
        for result in conversion_results:
            if result.get('success'):
                pdl_path = self.save_pdl_file(result)
                if pdl_path:
                    saved_files.append(pdl_path)
                    self.console.print(f"  [green]✓[/green] Saved: {Path(pdl_path).name}")
                else:
                    failed_files.append(Path(result['spf_file']).name)
            else:
                failed_files.append(Path(result['spf_file']).name)
        
        # Stage 4: Generate summary report
        self.console.print(f"\n[bold yellow]STAGE 4: Generating Summary[/bold yellow]", justify="center")
        
        summary = {
            "total_files_processed": len(spf_files),
            "successful_conversions": len(saved_files),
            "failed_conversions": len(failed_files),
            "total_ai_tokens_used": total_tokens,
            "converted_files": saved_files,
            "failed_files": failed_files,
            "conversion_results": conversion_results
        }
        
        # Save summary JSON
        summary_file = Path.cwd() / "spf_to_pdl_conversion_summary.json"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            self.console.print(f"\n[green]Summary saved to:[/green] [bold]{summary_file}[/bold]")
        except Exception as e:
            self.console.print(f"\n[red]Error saving summary: {e}[/red]")
        
        # Final summary display
        self.console.rule("[bold blue]Conversion Complete[/bold blue]", style="blue")
        self.console.print(f"""
[bold green]Summary:[/bold green]
  • Total SPF files: {len(spf_files)}
  • Successfully converted: {len(saved_files)}
  • Failed: {len(failed_files)}
  • AI Tokens used: {total_tokens}
        """, justify="center")
        
        if saved_files:
            self.console.print("\n[bold green]Converted PDL Files:[/bold green]")
            for pdl_file in saved_files:
                self.console.print(f"  • {pdl_file}")
        
        if failed_files:
            self.console.print("\n[bold red]Failed Conversions:[/bold red]")
            for failed_file in failed_files:
                self.console.print(f"  • {failed_file}")
