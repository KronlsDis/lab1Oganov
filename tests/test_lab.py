import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Импортируем mpi4py. Если он не установлен, обычные тесты
# всё равно смогут выполняться.
try:
    import mpi4py
    MPI4PY_AVAILABLE = True
except Exception:
    MPI4PY_AVAILABLE = False

    import types

    mpi4py_stub = types.ModuleType("mpi4py")
    mpi4py_stub.MPI = None
    sys.modules.setdefault("mpi4py", mpi4py_stub)


import sequential
import parallel_send_recv
import parallel_scatterv


# ----------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------

def write_input_files(directory, matrix, vector):
    """Создание входных файлов для тестирования MPI-программ."""

    data_dir = directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    m, n = matrix.shape

    (data_dir / "in.dat").write_text(
        f"{n}\n{m}\n",
        encoding="utf-8"
    )

    np.savetxt(
        data_dir / "AData.dat",
        matrix.reshape(-1),
        fmt="%.17g"
    )

    np.savetxt(
        data_dir / "xData.dat",
        vector,
        fmt="%.17g"
    )


def read_result(path):
    """Чтение результата работы параллельной программы."""

    assert path.exists(), f"Файл результата не создан: {path}"

    values = np.loadtxt(
        path,
        dtype=np.float64,
        ndmin=1
    )

    return np.asarray(values).reshape(-1)


# ----------------------------------------------------------------
# 1. Тестирование последовательной версии
# ----------------------------------------------------------------

def test_read_params_reads_N_then_M(tmp_path):
    path = tmp_path / "in.dat"
    path.write_text("5\n7\n", encoding="utf-8")

    assert sequential.read_params(path) == (5, 7)


def test_read_matrix_reads_values(tmp_path):
    path = tmp_path / "AData.dat"

    path.write_text(
        "1\n2\n3\n4\n5\n6\n",
        encoding="utf-8"
    )

    actual = sequential.read_matrix(
        path,
        M=2,
        N=3
    )

    np.testing.assert_allclose(
        actual,
        [[1, 2, 3], [4, 5, 6]]
    )


def test_read_vector_reads_values(tmp_path):
    path = tmp_path / "xData.dat"

    path.write_text(
        "1.5\n-2\n3.25\n",
        encoding="utf-8"
    )

    np.testing.assert_allclose(
        sequential.read_vector(path),
        [1.5, -2, 3.25]
    )


@pytest.mark.parametrize(
    "matrix, vector, expected",
    [
        ([[1, 2], [3, 4]], [5, 6], [17, 39]),
        ([[2]], [3], [6]),
        ([[0, 0], [0, 0]], [10, -5], [0, 0]),
        ([[-1, 2], [3, -4]], [2, -3], [-8, 18]),
        ([[1.25, 2.5]], [2, 4], [12.5]),
    ]
)
def test_sequential_mat_vec_mult_known_examples(
    matrix,
    vector,
    expected
):
    actual = sequential.sequential_mat_vec_mult(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(vector, dtype=np.float64)
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12
    )


@pytest.mark.parametrize(
    "m, n, seed",
    [
        (1, 1, 1),
        (2, 5, 2),
        (7, 5, 42),
        (10, 3, 99),
        (25, 25, 1234),
    ]
)
def test_sequential_result_matches_numpy_reference(m, n, seed):
    rng = np.random.default_rng(seed)

    matrix = rng.random((m, n))
    vector = rng.random(n)

    actual = sequential.sequential_mat_vec_mult(
        matrix,
        vector
    )

    expected = matrix @ vector

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12
    )


def test_read_params_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sequential.read_params(
            tmp_path / "missing.dat"
        )


def test_read_params_invalid_number_raises(tmp_path):
    path = tmp_path / "bad.dat"

    path.write_text(
        "not-a-number\n3\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        sequential.read_params(path)


# ----------------------------------------------------------------
# 2. Тестирование разбиения данных для Scatterv/Gatherv
# ----------------------------------------------------------------

@pytest.mark.parametrize(
    "m, processes, expected_counts, expected_displs",
    [
        (7, 4, [0, 3, 2, 2], [0, 0, 3, 5]),
        (8, 4, [0, 3, 3, 2], [0, 0, 3, 6]),
        (2, 4, [0, 1, 1, 0], [0, 0, 1, 2]),
        (1, 5, [0, 1, 0, 0, 0], [0, 0, 1, 1, 1]),
        (10, 3, [0, 5, 5], [0, 0, 5]),
        (0, 4, [0, 0, 0, 0], [0, 0, 0, 0]),
    ]
)
def test_scatterv_partition_arrays(
    m,
    processes,
    expected_counts,
    expected_displs
):
    counts, displs = (
        parallel_scatterv.auxiliary_arrays_determination(
            m,
            processes
        )
    )

    np.testing.assert_array_equal(
        counts,
        expected_counts
    )

    np.testing.assert_array_equal(
        displs,
        expected_displs
    )

    assert int(np.sum(counts)) == m
    assert counts.dtype == np.int32
    assert displs.dtype == np.int32


@pytest.mark.parametrize(
    "m, processes",
    [
        (7, 4),
        (2, 8),
        (1, 4),
        (200, 9),
        (0, 3),
    ]
)
def test_send_recv_partition_arrays_are_valid(m, processes):
    counts, displs = (
        parallel_send_recv.auxiliary_arrays_determination(
            m,
            processes
        )
    )

    assert len(counts) == processes
    assert len(displs) == processes

    assert counts[0] == 0
    assert displs[0] == 0

    assert int(np.sum(counts)) == m

    for rank in range(1, processes):
        assert displs[rank] == (
            displs[rank - 1] + counts[rank - 1]
        )

    assert (
        max(counts[1:], default=0)
        - min(counts[1:], default=0)
    ) <= 1


def test_partition_assigns_extra_rows_to_first_workers():
    counts, _ = (
        parallel_scatterv.auxiliary_arrays_determination(
            7,
            4
        )
    )

    assert counts.tolist() == [0, 3, 2, 2]


# ----------------------------------------------------------------
# 3. Тестирование генератора входных данных
# ----------------------------------------------------------------

def load_generator():
    generator_path = SCRIPTS / "generate_data.py"

    if not generator_path.exists():
        pytest.skip(
            f"Не найден генератор данных: {generator_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "generate_data",
        generator_path
    )

    module = importlib.util.module_from_spec(spec)

    assert spec is not None
    assert spec.loader is not None

    spec.loader.exec_module(module)

    return module


def test_generate_test_data_creates_three_files(tmp_path):
    generator = load_generator()

    output_dir = tmp_path / "generated"

    generator.generate_test_data(
        7,
        5,
        42,
        str(output_dir)
    )

    assert (output_dir / "in.dat").exists()
    assert (output_dir / "AData.dat").exists()
    assert (output_dir / "xData.dat").exists()

    dimensions = (
        output_dir / "in.dat"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    assert dimensions == ["5", "7"]

    matrix = np.loadtxt(
        output_dir / "AData.dat"
    ).reshape(7, 5)

    vector = np.loadtxt(
        output_dir / "xData.dat"
    )

    assert matrix.shape == (7, 5)
    assert vector.shape == (5,)

    assert np.all((matrix >= 0) & (matrix < 1))
    assert np.all((vector >= 0) & (vector < 1))


def test_generator_is_reproducible(tmp_path):
    generator = load_generator()

    first = tmp_path / "first"
    second = tmp_path / "second"

    generator.generate_test_data(
        4, 3, 2026, str(first)
    )

    generator.generate_test_data(
        4, 3, 2026, str(second)
    )

    for filename in (
        "in.dat",
        "AData.dat",
        "xData.dat"
    ):
        assert (
            (first / filename).read_text()
            ==
            (second / filename).read_text()
        )


# ----------------------------------------------------------------
# 4. MPI-интеграционные тесты
# ----------------------------------------------------------------

MPI_LAUNCHER = (
    shutil.which("mpiexec")
    or shutil.which("mpirun")
)

MPI_AVAILABLE = (
    MPI_LAUNCHER is not None
    and MPI4PY_AVAILABLE
)


def run_mpi_program(script, process_count, workdir):
    """Запуск MPI-программы в отдельном процессе."""

    command = [
        MPI_LAUNCHER,
        "-n",
        str(process_count),
        sys.executable,
        str(script)
    ]

    env = os.environ.copy()

    # Разрешение запуска Open MPI от root
    # в контейнерных и CI-средах.
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

    env["PYTHONPATH"] = (
        str(SRC)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    return subprocess.run(
        command,
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False
    )


@pytest.mark.skipif(
    not MPI_AVAILABLE,
    reason="Не найден mpiexec/mpirun или mpi4py"
)
@pytest.mark.parametrize(
    "m, n, processes",
    [
        (5, 3, 2),
        (6, 4, 3),
        (7, 5, 4),
        (2, 3, 8),
        (1, 1, 4),
        (8, 6, 4),
    ]
)
@pytest.mark.parametrize(
    "module_name",
    [
        "parallel_send_recv",
        "parallel_scatterv",
    ]
)
def test_mpi_program_matches_numpy(
    m,
    n,
    processes,
    module_name,
    tmp_path
):
    rng = np.random.default_rng(
        1000 + m * 10 + n
    )

    matrix = rng.normal(size=(m, n))
    vector = rng.normal(size=n)

    expected = matrix @ vector

    workdir = (
        tmp_path
        / f"{module_name}_{m}_{n}_{processes}"
    )

    workdir.mkdir()

    write_input_files(
        workdir,
        matrix,
        vector
    )

    script = SRC / f"{module_name}.py"

    result = run_mpi_program(
        script,
        processes,
        workdir
    )

    assert result.returncode == 0, (
        f"MPI-программа завершилась с кодом "
        f"{result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    actual = read_result(
        workdir / "Results_parallel.dat"
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-11
    )


@pytest.mark.skipif(
    not MPI_AVAILABLE,
    reason="Не найден mpiexec/mpirun"
)
def test_scatterv_reports_single_process_requirement(tmp_path):
    matrix = np.eye(2)
    vector = np.ones(2)

    write_input_files(
        tmp_path,
        matrix,
        vector
    )

    script = SRC / "parallel_scatterv.py"

    result = run_mpi_program(
        script,
        1,
        tmp_path
    )

    assert result.returncode == 0

    assert (
        "не менее 2 процессов"
        in result.stdout.lower()
    )


# ----------------------------------------------------------------
# 5. Проверка точности числовых результатов
# ----------------------------------------------------------------

def test_result_file_preserves_float_precision(tmp_path):
    path = tmp_path / "result.dat"

    expected = np.array([
        1 / 3,
        np.pi
    ])

    np.savetxt(
        path,
        expected,
        fmt="%.17g"
    )

    actual = read_result(path)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-15,
        atol=1e-15
    )
