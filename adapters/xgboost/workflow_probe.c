#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <dlfcn.h>

CAMLprim value modelkit_xgboost_test_openmp_limits(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  typedef int (*query)(void);
  query maximum = (query)dlsym(RTLD_DEFAULT, "omp_get_max_threads");
  query limit = (query)dlsym(RTLD_DEFAULT, "omp_get_thread_limit");
  if (maximum == NULL || limit == NULL) {
    caml_failwith("XGBoost test could not inspect the loaded OpenMP runtime");
  }
  int maximum_value = maximum();
  int limit_value = limit();
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(maximum_value));
  Store_field(result, 1, Val_int(limit_value));
  CAMLreturn(result);
}
