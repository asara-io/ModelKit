#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <dlfcn.h>

CAMLprim value modelkit_xgboost_test_openmp_limits(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  typedef int (*query)(void);
  void *program = dlopen(NULL, RTLD_LAZY);
  if (program == NULL) {
    caml_failwith("XGBoost test could not open the running program");
  }
  query maximum = (query)dlsym(program, "omp_get_max_threads");
  query limit = (query)dlsym(program, "omp_get_thread_limit");
  if (maximum == NULL || limit == NULL) {
    dlclose(program);
    caml_failwith("XGBoost test could not inspect the loaded OpenMP runtime");
  }
  int maximum_value = maximum();
  int limit_value = limit();
  dlclose(program);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(maximum_value));
  Store_field(result, 1, Val_int(limit_value));
  CAMLreturn(result);
}
