package io.vacco.jtinn.io;

import io.vacco.jtinn.Tinn;
import io.vacco.jtinn.XTinn;

import java.io.*;
import java.util.Arrays;
import java.util.Objects;

public class JtTextSerializer implements JtSerializer {

  private final PrintWriter out;

  public JtTextSerializer(PrintWriter out) {
    this.out = Objects.requireNonNull(out);
  }

  @Override public Tinn read(InputStream in) {
    try (BufferedReader r = new BufferedReader(new InputStreamReader(in))) {
      String header = r.readLine();
      int[] tData = Arrays.stream(header.split(" ")).mapToInt(Integer::parseInt).toArray();
      Tinn t = XTinn.xtbuild(tData[0], tData[1], tData[2]);
      for (int i = 0; i < t.nb; i++) { t.b[i] = Double.parseDouble(r.readLine()); }
      for (int i = 0; i < t.nw; i++) { t.w[i] = Double.parseDouble(r.readLine()); }
      for (int i = 0; i < t.x.length; i++) { t.x[i] = Double.parseDouble(r.readLine()); }
      in.close();
      return t;
    } catch (IOException ex) { throw new IllegalStateException(ex); }
  }

  @Override
  public void write(Tinn t) {
    out.printf("%d %d %d%n", t.nips, t.nhid, t.nops);
    for (double bias : t.b) { out.printf("%f%n", bias); }
    for (double hidWeight : t.w) { out.printf("%f%n", hidWeight); }
    for (double outWeight : t.x) { out.printf("%f%n", outWeight); }
    out.close();
  }

}
