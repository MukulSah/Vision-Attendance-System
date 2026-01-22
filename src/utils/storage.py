from pathlib import Path


def get_employee_face_dir(emp_id: str, emp_name: str) -> Path:
    """
    Returns:
    Documents/YOAR_AttendaceSystem/employees/{empId}_{empName}/face_data
    Creates directories if missing.
    """
    base = (
        Path.home()
        / "Documents"
        / "YOAR_AttendaceSystem"
        / "employees"
        / f"{emp_id}_{emp_name}"
        / "face_data"
    )

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_employees_root() -> Path:
    return (
        Path.home()
        / "Documents"
        / "YOAR_AttendaceSystem"
        / "employees"
    )
