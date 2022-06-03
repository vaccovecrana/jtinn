package io.vacco.jtinn;

import io.vacco.jtinn.util.JtPair;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.util.*;
import java.util.stream.Collectors;

import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtPairTest {
  static {
    it("Creates forward/backward pairs", () -> {
      List<String> letters = Arrays.asList("A", "B", "C", "D", "E");
      List<JtPair<String>> fwd = JtPair.pairsOfFwd(letters).collect(Collectors.toList());
      List<JtPair<String>> bkw = JtPair.pairOfRev(letters);

      System.out.println(fwd);
      System.out.println(bkw);
    });
  }
}
