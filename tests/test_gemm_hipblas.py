"""Tests for AMD GEMM HipBLASLt benchmark module."""

from benchmarks.amd import gemm_hipblas_lt


class TestParseHipblasResults:
    SAMPLE = (
        "transA,transB,grouped_gemm,batch_count,M,N,K,alpha,lda,beta,ldb,ldc,ldd,"
        "d_type,compute_type,activation_type,bias_vector,e_type,scaleA,scaleB,scaleC,"
        "scaleD,amaxD,scaleE,bias_source,rotating,algo_method,solution_index,kernel_index,"
        "gflops,gb_per_s,us_per_call\n"
        "T,N,0,1,1024,1024,1024,1,1024,0,1024,1024,1024,f16_r,c_f32_r,none,0,void,f32_r,"
        "f32_r,void,void,void,void,a,512,1,0,0,345670,123.45,6.78\n"
        "T,N,0,1,2048,2048,2048,1,2048,0,2048,2048,2048,f16_r,c_f32_r,none,0,void,f32_r,"
        "f32_r,void,void,void,void,a,512,1,0,0,567890,234.56,7.89\n"
    )

    def test_extracts_all_rows(self):
        rows = gemm_hipblas_lt.parse_hipblas_results(self.SAMPLE)
        assert len(rows) == 2

    def test_field_values(self):
        rows = gemm_hipblas_lt.parse_hipblas_results(self.SAMPLE)
        assert rows[0]["m"] == "1024"
        assert rows[0]["n"] == "1024"
        assert rows[0]["k"] == "1024"
        # fields[-3] is gflops: 345670, divided by 1000 = 345.67
        assert rows[0]["tflops"] == 345.67

    def test_second_row(self):
        rows = gemm_hipblas_lt.parse_hipblas_results(self.SAMPLE)
        assert rows[1]["m"] == "2048"
        assert rows[1]["tflops"] == 567.89

    def test_empty_input_returns_empty(self):
        assert gemm_hipblas_lt.parse_hipblas_results("") == []

    def test_no_t_lines_returns_empty(self):
        assert gemm_hipblas_lt.parse_hipblas_results("header line\nother line\n") == []

    def test_short_t_line_skipped(self):
        rows = gemm_hipblas_lt.parse_hipblas_results("T,N,0,1,1024\n")
        assert rows == []


class TestBuildTable:
    def test_smoke(self):
        rows = [{"m": "1024", "n": "1024", "k": "1024", "tflops": 345.67}]
        table = gemm_hipblas_lt._build_table(rows)
        text = table.get_string()
        assert "1024" in text
        assert "345.67" in text

    def test_empty_rows(self):
        table = gemm_hipblas_lt._build_table([])
        assert table.get_string() is not None


class TestModuleConstants:
    def test_image_constant(self):
        assert hasattr(gemm_hipblas_lt, "_HIPBLAS_IMAGE")

    def test_repo_constant(self):
        assert "hipBLASLt" in gemm_hipblas_lt._HIPBLASLT_REPO

    def test_run_is_callable(self):
        assert callable(gemm_hipblas_lt.run)
