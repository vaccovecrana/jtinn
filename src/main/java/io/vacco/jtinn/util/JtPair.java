package io.vacco.jtinn.util;

import java.util.*;
import java.util.stream.*;

public class JtPair<T> {

  public T e0, e1;

  public JtPair<T> swap() {
    JtPair<T> ps = new JtPair<>();
    ps.e0 = this.e1;
    ps.e1 = this.e0;
    return ps;
  }

  public static <T> JtPair<T> from(T e0, T e1) {
    JtPair<T> p = new JtPair<T>();
    p.e0 = Objects.requireNonNull(e0);
    p.e1 = Objects.requireNonNull(e1);
    return p;
  }

  public static <T> Stream<JtPair<T>> pairsOfFwd(List<T> in) {
    return IntStream.range(1, in.size())
        .mapToObj(i -> JtPair.from(in.get(i - 1), in.get(i)));
  }

  public static <T> List<JtPair<T>> pairOfRev(List<T> in) {
    List<JtPair<T>> fwd = pairsOfFwd(in)
        .map(JtPair::swap)
        .collect(Collectors.toList());
    Collections.reverse(fwd);
    return fwd;
  }

  @Override public String toString() { return String.format("(%s, %s)", e0, e1); }

}
