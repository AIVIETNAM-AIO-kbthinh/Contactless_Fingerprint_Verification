// Baseline SourceAFIS: đọc pairs.csv (probe,ref,label,same_subject), ghi scores.csv.
//
// Biên dịch / chạy (cần JDK ≥ 11 và jar SourceAFIS + phụ thuộc trong thư mục lib/):
//   javac -cp "lib/*" SourceAfisScores.java
//   java -cp "lib/*;." SourceAfisScores data/enhanced/pairs.csv data/enhanced/scores_sourceafis.csv 500
// (Linux/macOS dùng ':' thay cho ';' trong classpath.)
//
// Điểm SourceAFIS không nằm trong [0, 1]; metrics.py chỉ cần thứ tự nên dùng trực tiếp được:
//   python -m eval.baselines.score_csv data/enhanced/scores_sourceafis.csv

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class SourceAfisScores {
    private static final Map<String, FingerprintTemplate> CACHE = new HashMap<>();
    private static double dpi = 500;

    private static FingerprintTemplate template(String path) throws Exception {
        FingerprintTemplate t = CACHE.get(path);
        if (t == null) {
            byte[] bytes = Files.readAllBytes(Paths.get(path));
            t = new FingerprintTemplate(new FingerprintImage(bytes, new FingerprintImageOptions().dpi(dpi)));
            CACHE.put(path, t);
        }
        return t;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: SourceAfisScores pairs.csv scores.csv [dpi]");
            System.exit(2);
        }
        if (args.length >= 3) dpi = Double.parseDouble(args[2]);
        try (BufferedReader in = Files.newBufferedReader(Paths.get(args[0]));
             PrintWriter out = new PrintWriter(Files.newBufferedWriter(Paths.get(args[1])))) {
            String header = in.readLine();
            out.println("probe,ref,label,same_subject,score");
            String line;
            while ((line = in.readLine()) != null) {
                String[] f = line.split(",");
                double score = new FingerprintMatcher(template(f[0])).match(template(f[1]));
                out.printf("%s,%s,%s,%s,%.6f%n", f[0], f[1], f[2], f[3], score);
            }
        }
    }
}
