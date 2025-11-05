from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_group_html(group_row):
    """Create HTML for a single group with both tasks"""
    teilnehmer = group_row["Teilnehmer"]
    group_id = (
        str(teilnehmer)
        .replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )

    # Sentences task data
    sentences_html = group_row["sentences"] if pd.notna(group_row["sentences"]) else ""
    sentences_remarks = (
        group_row["sentences_remarks"]
        if pd.notna(group_row["sentences_remarks"])
        else ""
    )
    sentences_points = (
        float(group_row["sentences_points"])
        if pd.notna(group_row["sentences_points"])
        else 0.0
    )

    # Validity & Soundness task data
    validity_html = (
        group_row["validity_and_soundness"]
        if pd.notna(group_row["validity_and_soundness"])
        else ""
    )
    validity_remarks = (
        group_row["validity_and_soundness_remarks"]
        if pd.notna(group_row["validity_and_soundness_remarks"])
        else ""
    )
    validity_points = (
        float(group_row["validity_and_soundness_points"])
        if pd.notna(group_row["validity_and_soundness_points"])
        else 0.0
    )

    # Build sentences subtasks HTML
    sentences_subtasks_html = []
    for letter in ["a", "b", "c", "d", "e", "f"]:
        col_name = f"sentences_{letter}"
        subtask_content = group_row[col_name] if pd.notna(group_row[col_name]) else ""
        subtask_id = f"{group_id}_sentences_{letter}"
        subtask_html = f"""
        <div class="subtask-item">
            <div class="subtask-content">
                <strong>{letter}.</strong> {subtask_content}
            </div>
            <select class="grade-select" data-subtask="{subtask_id}" data-task="sentences">
                <option value="0">0</option>
                <option value="0.5">0.5</option>
                <option value="1" selected>1</option>
            </select>
        </div>
        """
        sentences_subtasks_html.append(subtask_html)

    # Build validity subtasks HTML
    validity_subtasks_html = []
    for letter in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        col_name = f"validity_and_soundness_{letter}"
        subtask_content = group_row[col_name] if pd.notna(group_row[col_name]) else ""
        subtask_id = f"{group_id}_validity_{letter}"
        subtask_html = f"""
        <div class="subtask-item">
            <div class="subtask-content">
                <strong>{letter}.</strong> {subtask_content}
            </div>
            <select class="grade-select" data-subtask="{subtask_id}" data-task="validity">
                <option value="0">0</option>
                <option value="0.5">0.5</option>
                <option value="1" selected>1</option>
            </select>
        </div>
        """
        validity_subtasks_html.append(subtask_html)

    # Calculate expected totals
    sentences_total = sum([1.0] * 6)  # Default all to 1
    validity_total = sum([1.0] * 8)  # Default all to 1

    group_html = f"""
    <details class="group-container" data-group-id="{group_id}" data-teilnehmer="{teilnehmer}" open>
        <summary>
            <strong>{teilnehmer}</strong>
            <span class="group-totals">
                (Sentences: <span id="{group_id}_sentences_sum">0</span>/{sentences_points}, 
                Validity: <span id="{group_id}_validity_sum">0</span>/{validity_points})
            </span>
        </summary>
        
        <div class="task-section">
            <h3>SENTENCES <span class="task-sum">(Sum: <span id="{group_id}_sentences_display">0</span>/{sentences_points})</span></h3>
            <div class="task-content">
                <div class="full-task">
                    <h4>Full Task:</h4>
                    <div class="html-content">{sentences_html}</div>
                </div>
                <div class="remarks">
                    <h4>Remarks:</h4>
                    <div class="html-content">{sentences_remarks}</div>
                </div>
                <div class="subtasks">
                    <h4>Subtasks:</h4>
                    {"".join(sentences_subtasks_html)}
                </div>
            </div>
        </div>
        
        <div class="task-section">
            <h3>VALIDITY & SOUNDNESS <span class="task-sum">(Sum: <span id="{group_id}_validity_display">0</span>/{validity_points})</span></h3>
            <div class="task-content">
                <div class="full-task">
                    <h4>Full Task:</h4>
                    <div class="html-content">{validity_html}</div>
                </div>
                <div class="remarks">
                    <h4>Remarks:</h4>
                    <div class="html-content">{validity_remarks}</div>
                </div>
                <div class="subtasks">
                    <h4>Subtasks:</h4>
                    {"".join(validity_subtasks_html)}
                </div>
            </div>
        </div>
    </details>
    """

    return group_html


def generate_html():
    """Generate the complete HTML grading interface"""
    csv_path = PROJECT_ROOT / "data" / "logic" / "quiz_1_subtasks.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    # Generate HTML for each group
    groups_html = []
    for _, row in df.iterrows():
        groups_html.append(create_group_html(row))

    # Create the complete HTML document
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logic Quiz Grading Interface</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        
        .export-section {{
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .export-btn {{
            padding: 12px 24px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            font-weight: bold;
        }}
        
        .export-btn:hover {{
            background-color: #0056b3;
        }}
        
        details {{
            background-color: white;
            margin: 15px 0;
            padding: 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        summary {{
            cursor: pointer;
            font-weight: bold;
            padding: 15px 20px;
            background-color: #e9ecef;
            border-radius: 8px 8px 0 0;
            font-size: 16px;
        }}
        
        summary:hover {{
            background-color: #dee2e6;
        }}
        
        .group-totals {{
            float: right;
            font-weight: normal;
            color: #666;
        }}
        
        .task-section {{
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .task-section:last-child {{
            border-bottom: none;
        }}
        
        .task-section h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        
        .task-sum {{
            font-size: 14px;
            font-weight: normal;
            color: #666;
        }}
        
        .task-content {{
            margin-top: 15px;
        }}
        
        .full-task, .remarks {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        
        .full-task h4, .remarks h4 {{
            margin-top: 0;
            color: #495057;
        }}
        
        .html-content {{
            line-height: 1.6;
            color: #212529;
        }}
        
        .subtasks {{
            margin-top: 20px;
        }}
        
        .subtasks h4 {{
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .subtask-item {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 12px;
            margin-bottom: 10px;
            background-color: #f8f9fa;
            border-left: 3px solid #007bff;
            border-radius: 4px;
        }}
        
        .subtask-content {{
            flex: 1;
            margin-right: 15px;
            line-height: 1.6;
        }}
        
        .grade-select {{
            padding: 6px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
            min-width: 70px;
            background-color: white;
        }}
        
        .grade-select:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
        }}
    </style>
</head>
<body>
    <h1>Logic Quiz Grading Interface</h1>
    
    <div class="export-section">
        <button class="export-btn" id="export-btn">Export Grades to JSON</button>
    </div>
    
    {"".join(groups_html)}
    
    <script>
        // Initialize sums on page load
        document.addEventListener('DOMContentLoaded', function() {{
            loadSavedGrades();
            
            // Add event listener to export button
            const exportBtn = document.getElementById('export-btn');
            if (exportBtn) {{
                exportBtn.addEventListener('click', exportGrades);
            }}
        }});
        
        // Add event listeners to all grade selects
        document.addEventListener('change', function(e) {{
            if (e.target.classList.contains('grade-select')) {{
                const subtaskId = e.target.getAttribute('data-subtask');
                const grade = parseFloat(e.target.value);
                
                // Save to localStorage
                localStorage.setItem(`grade_${{subtaskId}}`, grade);
                
                // Update sums
                updateSums(e.target);
            }}
        }});
        
        function updateSums(selectElement) {{
            const groupId = selectElement.closest('details').getAttribute('data-group-id');
            const task = selectElement.getAttribute('data-task');
            
            // Get all selects for this task in this group
            const selects = document.querySelectorAll(
                `details[data-group-id="${{groupId}}"] select[data-task="${{task}}"]`
            );
            
            let sum = 0;
            selects.forEach(select => {{
                sum += parseFloat(select.value);
            }});
            
            // Update display
            const displayId = task === 'sentences' 
                ? `${{groupId}}_sentences_display` 
                : `${{groupId}}_validity_display`;
            const summaryId = task === 'sentences'
                ? `${{groupId}}_sentences_sum`
                : `${{groupId}}_validity_sum`;
            
            document.getElementById(displayId).textContent = sum.toFixed(1);
            document.getElementById(summaryId).textContent = sum.toFixed(1);
        }}
        
        function updateAllSums() {{
            const allSelects = document.querySelectorAll('.grade-select');
            const groups = new Set();
            
            allSelects.forEach(select => {{
                const groupId = select.closest('details').getAttribute('data-group-id');
                groups.add(groupId);
            }});
            
            groups.forEach(groupId => {{
                const sentencesSelects = document.querySelectorAll(
                    `details[data-group-id="${{groupId}}"] select[data-task="sentences"]`
                );
                const validitySelects = document.querySelectorAll(
                    `details[data-group-id="${{groupId}}"] select[data-task="validity"]`
                );
                
                if (sentencesSelects.length > 0) {{
                    updateSums(sentencesSelects[0]);
                }}
                if (validitySelects.length > 0) {{
                    updateSums(validitySelects[0]);
                }}
            }});
        }}
        
        // Initialize sums on page load
        document.addEventListener('DOMContentLoaded', function() {{
            loadSavedGrades();
        }});
        
        function loadSavedGrades() {{
            const allSelects = document.querySelectorAll('.grade-select');
            allSelects.forEach(select => {{
                const subtaskId = select.getAttribute('data-subtask');
                const savedGrade = localStorage.getItem(`grade_${{subtaskId}}`);
                if (savedGrade !== null) {{
                    select.value = savedGrade;
                }}
            }});
            // Update sums after loading saved grades
            updateAllSums();
        }}
        
        function exportGrades() {{
            const grades = {{}};
            const allSelects = document.querySelectorAll('.grade-select');
            
            allSelects.forEach(select => {{
                const subtaskId = select.getAttribute('data-subtask');
                const grade = parseFloat(select.value);
                
                // Get the Teilnehmer name from the parent details element
                const detailsElement = select.closest('details');
                const teilnehmer = detailsElement.getAttribute('data-teilnehmer');
                
                // Parse subtask type from subtaskId
                const parts = subtaskId.split('_');
                const taskType = parts[1];
                const letter = parts[2];
                
                if (!grades[teilnehmer]) {{
                    grades[teilnehmer] = {{}};
                }}
                
                if (taskType === 'sentences') {{
                    grades[teilnehmer][`sentences_${{letter}}`] = grade;
                }} else if (taskType === 'validity') {{
                    grades[teilnehmer][`validity_and_soundness_${{letter}}`] = grade;
                }}
            }});
            
            // Convert to JSON and download
            const jsonStr = JSON.stringify(grades, null, 2);
            const blob = new Blob([jsonStr], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'grading_results.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('Grades exported successfully!');
        }}
    </script>
</body>
</html>
    """

    # Save HTML file
    output_path = PROJECT_ROOT / "grading_interface.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Grading interface generated: {output_path}")
    print(f"Total groups: {len(df)}")


if __name__ == "__main__":
    generate_html()
