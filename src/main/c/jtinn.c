#include <jni.h>
#include <immintrin.h>
#include <stdlib.h>

static float hsum256_ps(__m256 v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  __m128 vsum = _mm_hadd_ps(vlow, vlow);
  vsum = _mm_hadd_ps(vsum, vsum);
  return _mm_cvtss_f32(vsum);
}

JNIEXPORT void JNICALL Java_io_vacco_jtinn_JtVec_avxFloatMatMul(
    JNIEnv *env, jclass cls,
    jfloatArray in, jobjectArray w, jfloatArray b, jfloatArray out,
    jint inSize, jint outSize
) {
  jfloat *inPtr = (*env)->GetPrimitiveArrayCritical(env, in, NULL);
  jfloat *bPtr = NULL;
  if (b != NULL) {
    bPtr = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
  }
  jfloat *outPtr = (*env)->GetPrimitiveArrayCritical(env, out, NULL);

  jfloat **wPtr = (jfloat **) malloc(outSize * sizeof(jfloat *));
  for (jint j = 0; j < outSize; j++) {
    jfloatArray row = (jfloatArray) (*env)->GetObjectArrayElement(env, w, j);
    wPtr[j] = (*env)->GetPrimitiveArrayCritical(env, row, NULL);
  }

  for (jint j = 0; j < outSize; j++) {
    __m256 acc = _mm256_setzero_ps();
    jint a;
    for (a = 0; a + 8 <= inSize; a += 8) {
      __m256 inVec = _mm256_loadu_ps(&inPtr[a]);
      __m256 wVec = _mm256_loadu_ps(&wPtr[j][a]);
      acc = _mm256_fmadd_ps(inVec, wVec, acc);
    }
    float sum = hsum256_ps(acc);
    for (; a < inSize; a++) {
      sum += inPtr[a] * wPtr[j][a];
    }
    float z = sum;
    if (bPtr != NULL) {
      z += bPtr[j];
    }
    outPtr[j] = z;
  }

  for (jint j = 0; j < outSize; j++) {
    jfloatArray row = (jfloatArray) (*env)->GetObjectArrayElement(env, w, j);
    (*env)->ReleasePrimitiveArrayCritical(env, row, wPtr[j], 0);
  }
  free(wPtr);

  (*env)->ReleasePrimitiveArrayCritical(env, out, outPtr, 0);
  if (bPtr != NULL) {
    (*env)->ReleasePrimitiveArrayCritical(env, b, bPtr, 0);
  }
  (*env)->ReleasePrimitiveArrayCritical(env, in, inPtr, 0);
}

static int hsum128_epi32(__m128i v) {
  __m128i h1 = _mm_hadd_epi32(v, v);
  __m128i h2 = _mm_hadd_epi32(h1, h1);
  return _mm_cvtsi128_si32(h2);
}

JNIEXPORT void JNICALL Java_io_vacco_jtinn_JtVec_sseInt8MatMul(
    JNIEnv *env, jclass cls,
    jbyteArray in, jobjectArray w, jfloatArray b, jfloatArray out,
    jint inSize, jint outSize, jfloat scale
) {
  jbyte *inPtr = (*env)->GetPrimitiveArrayCritical(env, in, NULL);
  jfloat *bPtr = NULL;
  if (b != NULL) {
    bPtr = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
  }
  jfloat *outPtr = (*env)->GetPrimitiveArrayCritical(env, out, NULL);

  jbyte **wPtr = (jbyte **) malloc(outSize * sizeof(jbyte *));
  for (jint j = 0; j < outSize; j++) {
    jbyteArray row = (jbyteArray) (*env)->GetObjectArrayElement(env, w, j);
    wPtr[j] = (*env)->GetPrimitiveArrayCritical(env, row, NULL);
  }

  for (jint j = 0; j < outSize; j++) {
    __m128i acc_low = _mm_setzero_si128();
    __m128i acc_high = _mm_setzero_si128();
    jint a;
    for (a = 0; a + 16 <= inSize; a += 16) {
      __m128i inVec = _mm_loadu_si128((__m128i *)&inPtr[a]);
      __m128i wVec = _mm_loadu_si128((__m128i *)&wPtr[j][a]);
      __m128i madd = _mm_maddubs_epi16(inVec, wVec);
      __m128i low = _mm_cvtepi16_epi32(madd);
      __m128i high = _mm_cvtepi16_epi32(_mm_srli_si128(madd, 8));
      acc_low = _mm_add_epi32(acc_low, low);
      acc_high = _mm_add_epi32(acc_high, high);
    }
    __m128i acc = _mm_add_epi32(acc_low, acc_high);
    int sum = hsum128_epi32(acc);
    for (; a < inSize; a++) {
      sum += (int)inPtr[a] * (int)wPtr[j][a];
    }
    float z = sum * scale;
    if (bPtr != NULL) {
      z += bPtr[j];
    }
    outPtr[j] = z;
  }

  for (jint j = 0; j < outSize; j++) {
    jbyteArray row = (jbyteArray) (*env)->GetObjectArrayElement(env, w, j);
    (*env)->ReleasePrimitiveArrayCritical(env, row, wPtr[j], 0);
  }
  free(wPtr);

  (*env)->ReleasePrimitiveArrayCritical(env, out, outPtr, 0);
  if (bPtr != NULL) {
    (*env)->ReleasePrimitiveArrayCritical(env, b, bPtr, 0);
  }
  (*env)->ReleasePrimitiveArrayCritical(env, in, inPtr, 0);
}
