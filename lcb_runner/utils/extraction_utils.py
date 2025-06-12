from lcb_runner.lm_styles import LMStyle


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()
    else:
        try:
            sol_idx = next(i for i, line in enumerate(outputlines)
                       if "solution code" in line.lower())
        except StopIteration:
            sol_idx = None
        # graceful fallback if the marker is missing 
        if sol_idx is None:
            # just grab the first fenced block in the whole output
            fences = [i for i, l in enumerate(outputlines) if "```" in l]
            if len(fences) >= 2:
                start_mark, end_mark = fences[0], fences[1]
            else:
                return ""
        else:
            # find the first and second ``` after the marker 
            start_mark = end_mark = None
            for i in range(sol_idx + 1, len(outputlines)):
                if "```" in outputlines[i]:
                    if start_mark is None:
                        start_mark = i
                    else:
                        end_mark = i
                        break
            if start_mark is None or end_mark is None or end_mark <= start_mark:
                return ""
        # extract & clean the block 
        code_lines = outputlines[start_mark + 1 : end_mark]
        if code_lines and code_lines[0].strip().lower() == "python":
            code_lines = code_lines[1:]
        # Drop any stray </answer> that some finetunes append
        if code_lines and code_lines[-1].strip().lower().startswith("</answer"):
            code_lines = code_lines[:-1]
        return "\n".join(code_lines)


def extract_test_output_code(model_output: str, lmstyle: LMStyle = None):
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        # first try to extract ```python if not then try ```
        indexlines = [
            i
            for i, line in enumerate(outputlines)
            if "```python" in line or "```Python" in line
        ]
        if indexlines:
            start_index = indexlines[0]
        else:
            start_index = None
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if start_index is not None:
            indexlines = [i for i in indexlines if i > start_index]
            indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, lmstyle: LMStyle, cot: bool = False):
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()