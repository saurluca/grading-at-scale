# %%
from pathlib import Path
import pandas as pd
import re
import html


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


df_raw = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_raw.csv", sep=";")

df_tasks = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_tasks.csv", sep=";")

df_points = pd.read_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_points.csv", sep=";")

df_remarks = pd.read_csv(
    PROJECT_ROOT / "data" / "logic" / "quiz_1_remarks.csv", sep=";"
)

# %%


def merge_raw_points_remarks(df_raw, df_points, df_remarks):
    # goal is to merge all df into one dataframe, but first correct tasks and columns need to be selected

    # Columns of df_raw: Index(['Teilnehmer', '1. Sentences', '2. Validity and Soundness (I)',
    #        '3. Validity and Soundness (II)', '4. Arguments'],
    #       dtype='object')
    # Columns of df_points: Index(['Teilnehmer', '1. Sentences', '2. Validity and Soundness (I)',
    #        '3. Validity and Soundness (II)', '4. Arguments', 'Summe'],
    #       dtype='object')
    # Columns of df_remarks: Index(['task_id', 'username', 'remark_html'], dtype='object')
    # Columns of df_tasks: Index(['question', 'answer', 'task'], dtype='object')

    # First rename in df_points, "1. Sentences" to "sentences_points", and "3. Validity and Soundness (II)" to "validity_and_soundness_points"
    df_points = df_points.rename(
        columns={
            "1. Sentences": "sentences_points",
            "3. Validity and Soundness (II)": "validity_and_soundness_points",
        }
    )

    # rename columns of df_raw to match df_points
    df_raw = df_raw.rename(
        columns={
            "1. Sentences": "sentences",
            "3. Validity and Soundness (II)": "validity_and_soundness",
        }
    )

    # drop columns "2. Validity and Soundness (I)" and "4. Arguments" from df_raw and df_points
    df_raw = df_raw.drop(columns=["2. Validity and Soundness (I)", "4. Arguments"])
    df_points = df_points.drop(
        columns=["2. Validity and Soundness (I)", "4. Arguments", "Summe", "Teilnehmer"]
    )

    # first we concat df_raw and df_points along the Teilnehmer column
    df_concat = pd.concat([df_raw, df_points], axis=1, join="inner")

    # assert length of both orgnal dfs and of df_concat same
    assert len(df_raw) == len(df_points) == len(df_concat)

    # df_remarks has columns ['task_id', 'username', 'remark_html'], change task_id 0 to sentences and 2 to validity_and_soundness
    df_remarks["task_id"] = df_remarks["task_id"].replace(
        {0: "sentences", 2: "validity_and_soundness"}
    )

    # merge df_concat and df_remarks, make two columsn from remarks. sentences_remarks and validity_and_soundness_remarks
    # pivot df_remarks so that task_id values become column names
    df_remarks_pivot = df_remarks.pivot(
        index="username", columns="task_id", values="remark_html"
    ).reset_index()

    # rename columns to add _remarks suffix
    df_remarks_pivot = df_remarks_pivot.rename(
        columns={
            "sentences": "sentences_remarks",
            "validity_and_soundness": "validity_and_soundness_remarks",
        }
    )

    # merge with df_concat on Teilnehmer and username
    df_concat = df_concat.merge(
        df_remarks_pivot, left_on="Teilnehmer", right_on="username", how="left"
    )

    # drop the username column from the merge (it's redundant with Teilnehmer)
    df_concat = df_concat.drop(columns=["username"])

    return df_concat


df_concat = merge_raw_points_remarks(df_raw, df_points, df_remarks)

df = df_concat.copy()
df_concat.head()


# %%


# print dtype of all columns
df.dtypes

# convert Teilnehmer, sentences, validity_and_soundness, sentences_remarks, validity_and_soundness_remarks, to string
df["Teilnehmer"] = df["Teilnehmer"].astype(str)
df["sentences"] = df["sentences"].astype(str)
df["validity_and_soundness"] = df["validity_and_soundness"].astype(str)
df["sentences_remarks"] = df["sentences_remarks"].astype(str)
df["validity_and_soundness_remarks"] = df["validity_and_soundness_remarks"].astype(str)

# Convert German-style numbers with commas to points, then to float
df["sentences_points"] = df["sentences_points"].str.replace(",", ".").astype(float)
df["validity_and_soundness_points"] = (
    df["validity_and_soundness_points"].str.replace(",", ".").astype(float)
)

df.dtypes

# %%
# next goal is to clean up the html of sentences and validity_and_soundness tasks


# remove all background color spans (even if they only have background-color style) for all colors with regex
def remove_background_colors(text):
    # Remove all <span style="background-color:rgb(...); ..."> tags (catch any style after or not)
    text = re.sub(
        r'<span style="[^"]*background-color:rgb\(\d{1,3},\d{1,3},\d{1,3}\);?[^"]*">',
        "",
        text,
    )
    # Also remove <span style="background-color:transparent;color:#000000;">
    text = re.sub(
        r'<span style="?background-color:transparent;color:#000000;"?>',
        "",
        text,
    )
    # Also remove <span style="color:rgb(0,0,0);">
    text = re.sub(
        r'<span style="?color:rgb\(0,0,0\);"?>',
        "",
        text,
    )
    return text


df["sentences"] = df["sentences"].apply(remove_background_colors)
df["validity_and_soundness"] = df["validity_and_soundness"].apply(
    remove_background_colors
)

# remove <!--HTML-->
df["sentences"] = df["sentences"].str.replace("<!--HTML-->", "")
df["validity_and_soundness"] = df["validity_and_soundness"].str.replace(
    "<!--HTML-->", ""
)


# replace </p><p> with <br /> for uniform parsing
df["sentences"] = df["sentences"].str.replace("</p><p>", "<br />")
df["validity_and_soundness"] = df["validity_and_soundness"].str.replace(
    "</p><p>", "<br />"
)

# replace <br /><br /> with just <br />
df["sentences"] = df["sentences"].str.replace("<br /><br />", "<br />")
df["validity_and_soundness"] = df["validity_and_soundness"].str.replace(
    "<br /><br />", "<br />"
)

df.head()

# %%

# next up, clean up html of sentences_remarks and validity_and_soundness_remarks


# Function to decode HTML entities, handling multiple levels of encoding
def decode_html_entities(text):
    """
    Decode HTML entities, handling nested encodings like &amp;lt; -> &lt; -> <
    """
    if pd.isna(text):
        return text

    # Convert to string if not already
    text = str(text)

    # Keep decoding until no more changes occur (handles nested encodings)
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = html.unescape(text)

    return text


# Apply HTML entity decoding to remarks columns
df["sentences_remarks"] = df["sentences_remarks"].apply(decode_html_entities)
df["validity_and_soundness_remarks"] = df["validity_and_soundness_remarks"].apply(
    decode_html_entities
)

# replace </p><p> with <br /> for uniform parsing
df["sentences_remarks"] = df["sentences_remarks"].str.replace("</p><p>", "<br />")
df["validity_and_soundness_remarks"] = df["validity_and_soundness_remarks"].str.replace(
    "</p><p>", "<br />"
)

# %%
# save df_concat to csv
df.to_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_cleaned.csv", index=False, sep=";")

df.head()

# %%

# next up, split sentences and validity_and_soundness into subtasks, as extra columns.
# for sentences, there are 6 subtasks, for validity_and_soundness, there are 8 subtasks, indicated by a., b., c., d., e., f., g., h.


def extract_subtask(text, letter):
    """
    Extract content for a specific subtask letter (a-h) from text.
    Pattern: letter. followed by content until next letter. or end
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Pattern to match the specific letter subtask: letter. followed by content
    # Capture everything (including HTML) until next letter pattern or end
    # Use negative lookahead to stop at next letter pattern (a-h followed by period)
    pattern = rf"{letter}\.\s*((?:(?!\b[a-h]\.).)*)"

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Clean up any trailing HTML closing tags or whitespace
        content = re.sub(r"\s*</?p>\s*$", "", content)
        content = re.sub(r"\s*<br\s*/?>\s*$", "", content)
        return content.strip()

    return ""


# Extract subtasks for sentences (a-f, 6 subtasks)
for letter in ["a", "b", "c", "d", "e", "f"]:
    df[f"sentences_{letter}"] = df["sentences"].apply(
        lambda x: extract_subtask(x, letter)
    )

# Extract subtasks for validity_and_soundness (a-h, 8 subtasks)
for letter in ["a", "b", "c", "d", "e", "f", "g", "h"]:
    df[f"validity_and_soundness_{letter}"] = df["validity_and_soundness"].apply(
        lambda x: extract_subtask(x, letter)
    )

# %%

df.head()

# %%

# save df to csv
df.to_csv(PROJECT_ROOT / "data" / "logic" / "quiz_1_subtasks.csv", index=False, sep=";")

# %%
