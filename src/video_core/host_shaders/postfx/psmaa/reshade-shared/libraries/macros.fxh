#ifndef _MACROS_FXH
#define _MACROS_FXH

// Base recursive macro helpers
#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

#define VAR_PREFIX x

#define VAR(prefx, nr) CONCAT(prefx, nr)
#define VAR_DEF(nr) VAR(VAR_PREFIX, nr)

#define PARAMS_1(type, prefx) \
  type VAR(prefx, 1)

#define PARAMS_2(type, prefx) \
  PARAMS_1(type, prefx), type VAR(prefx, 2)

#define PARAMS_3(type, prefx) \
  PARAMS_2(type, prefx), type VAR(prefx, 3)

#define PARAMS_4(type, prefx) \
  PARAMS_3(type, prefx), type VAR(prefx, 4)

#define PARAMS_5(type, prefx) \
  PARAMS_4(type, prefx), type VAR(prefx, 5)

#define PARAMS_6(type, prefx) \
  PARAMS_5(type, prefx), type VAR(prefx, 6)

#define PARAMS_7(type, prefx) \
  PARAMS_6(type, prefx), type VAR(prefx, 7)

#define PARAMS_8(type, prefx) \
  PARAMS_7(type, prefx), type VAR(prefx, 8)

#define PARAMS_9(type, prefx) \
  PARAMS_8(type, prefx), type VAR(prefx, 9)

#define PARAMS_10(type, prefx) \
  PARAMS_9(type, prefx), type VAR(prefx, 10)

#define PARAMS_11(type, prefx) \
  PARAMS_10(type, prefx), type VAR(prefx, 11)

#define PARAMS_12(type, prefx) \
  PARAMS_11(type, prefx), type VAR(prefx, 12)

#define PARAMS_13(type, prefx) \
  PARAMS_12(type, prefx), type VAR(prefx, 13)

#define PARAMS_14(type, prefx) \
  PARAMS_13(type, prefx), type VAR(prefx, 14)

#define PARAMS_15(type, prefx) \
  PARAMS_14(type, prefx), type VAR(prefx, 15)

#define PARAMS_16(type, prefx) \
  PARAMS_15(type, prefx), type VAR(prefx, 16)

#define PARAMS_DEF(type, n) \
  CONCAT(PARAMS_, n)(type, VAR_PREFIX)

#define GEN_FUNC_3_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 3)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2)) ,VAR_DEF(3)); \
  }

#define GEN_FUNC_4_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 4)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2)), func_name(VAR_DEF(3), VAR_DEF(4))); \
  }

#define GEN_FUNC_5_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 5)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2)), func_name(VAR_DEF(3), VAR_DEF(4), VAR_DEF(5))); \
  }

#define GEN_FUNC_6_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 6)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2)), func_name(VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6))); \
  }

#define GEN_FUNC_7_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 7)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3)), func_name(VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7))); \
  }

#define GEN_FUNC_8_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 8)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4)), func_name(VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8))); \
  }

#define GEN_FUNC_9_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 9)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4)), func_name(VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8), VAR_DEF(9))); \
  }

#define GEN_FUNC_10_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 10)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6)), func_name(VAR_DEF(7), VAR_DEF(8), VAR_DEF(9), VAR_DEF(10))); \
  }

#define GEN_FUNC_11_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 11)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7)), func_name(VAR_DEF(8), VAR_DEF(9), VAR_DEF(10), VAR_DEF(11))); \
  }

#define GEN_FUNC_12_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 12)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8)), func_name(VAR_DEF(9), VAR_DEF(10), VAR_DEF(11), VAR_DEF(12))); \
  }

#define GEN_FUNC_13_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 13)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8)), func_name(VAR_DEF(9), VAR_DEF(10), VAR_DEF(11), VAR_DEF(12), VAR_DEF(13))); \
  }

#define GEN_FUNC_14_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 14)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8)), func_name(VAR_DEF(9), VAR_DEF(10), VAR_DEF(11), VAR_DEF(12), VAR_DEF(13), VAR_DEF(14))); \
  }

#define GEN_FUNC_15_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 15)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8)), func_name(VAR_DEF(9), VAR_DEF(10), VAR_DEF(11), VAR_DEF(12), VAR_DEF(13), VAR_DEF(14), VAR_DEF(15))); \
  }

#define GEN_FUNC_16_PARAMS(type, func_name) \
  type func_name(PARAMS_DEF(type, 16)) \
  { \
    return func_name(func_name(VAR_DEF(1), VAR_DEF(2), VAR_DEF(3), VAR_DEF(4), VAR_DEF(5), VAR_DEF(6), VAR_DEF(7), VAR_DEF(8)), func_name(VAR_DEF(9), VAR_DEF(10), VAR_DEF(11), VAR_DEF(12), VAR_DEF(13), VAR_DEF(14), VAR_DEF(15), VAR_DEF(16))); \
  }
#endif // MACROS_FXH include guard
