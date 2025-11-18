import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;


// using TF-IDF + Cosine Similarity for intent matching.

public class ChatbotTFIDF {

    private static final double THRESHOLD = 0.15;
    private static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
            "a","an","the","is","are","was","were","in","on","at","for","to","and","or","of","i","you","me","my","your"
    ));

    private static class Intent {
        String name;
        List<String> patterns = new ArrayList<>();
        List<String> responses = new ArrayList<>();
        double[] vector; 
    }

    private final List<Intent> intents = new ArrayList<>();
    private final Random rnd = new Random();


    private final Map<String, Integer> vocabIndex = new HashMap<>();
    private double[] idf; 
    private int totalDocs = 0; 

    public ChatbotTFIDF(String intentsFilePath) throws IOException {
        loadIntents(intentsFilePath);
        buildVocabularyAndVectors();
    }


    private void loadIntents(String pathStr) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(pathStr));
        Intent cur = null;
        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty()) {
                if (cur != null) intents.add(cur);
                cur = null;
                continue;
            }
            if (line.startsWith("#")) {
                cur = new Intent();
                cur.name = line.substring(1).trim();
                continue;
            }
            if (cur == null) continue;
            if (line.startsWith("patterns:")) {
                String after = line.substring("patterns:".length()).trim();
                if (!after.isEmpty()) {
                    String[] parts = after.split("\\|");
                    for (String p : parts) {
                        String pp = p.trim();
                        if (!pp.isEmpty()) cur.patterns.add(pp);
                    }
                }
            } else if (line.startsWith("responses:")) {
                String after = line.substring("responses:".length()).trim();
                if (!after.isEmpty()) {
                    String[] parts = after.split("\\|");
                    for (String r : parts) {
                        String rr = r.trim();
                        if (!rr.isEmpty()) cur.responses.add(rr);
                    }
                }
            }
        }
        if (cur != null) intents.add(cur);
    }

    private void buildVocabularyAndVectors() {
        
        List<List<String>> documents = new ArrayList<>();
        List<Intent> patternIntentMap = new ArrayList<>(); 
        for (Intent intent : intents) {
            for (String pattern : intent.patterns) {
                List<String> tokens = tokenize(pattern);
                if (tokens.isEmpty()) continue;
                documents.add(tokens);
                patternIntentMap.add(intent);
            }
        }

        totalDocs = documents.size();
        Map<String, Integer> df = new HashMap<>();
        for (List<String> doc : documents) {
            Set<String> seen = new HashSet<>(doc);
            for (String t : seen) df.put(t, df.getOrDefault(t, 0) + 1);
        }

        int idx = 0;
        for (String term : df.keySet().stream().sorted().collect(Collectors.toList())) {
            vocabIndex.put(term, idx++);
        }

        idf = new double[vocabIndex.size()];
        for (Map.Entry<String, Integer> e : df.entrySet()) {
            String term = e.getKey();
            int docFreq = e.getValue();
            int i = vocabIndex.get(term);
            idf[i] = Math.log((double)(totalDocs + 1) / (docFreq + 1)) + 1.0;
        }

        List<double[]> docVectors = new ArrayList<>();
        for (List<String> doc : documents) {
            double[] vec = new double[vocabIndex.size()];
            Map<String, Integer> counts = new HashMap<>();
            int maxCount = 0;
            for (String t : doc) {
                int c = counts.getOrDefault(t, 0) + 1;
                counts.put(t, c);
                if (c > maxCount) maxCount = c;
            }
            
            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                String term = e.getKey();
                int c = e.getValue();
                Integer pos = vocabIndex.get(term);
                if (pos == null) continue;
                double tf = (double) c / (double) maxCount; 
                vec[pos] = tf * idf[pos];
            }
            normalizeInPlace(vec);
            docVectors.add(vec);
        }

        Map<Intent, List<double[]>> intentToDocVectors = new HashMap<>();
        int docIndex = 0;
        for (List<String> doc : documents) {
            Intent correspondingIntent = patternIntentMap.get(docIndex);
            intentToDocVectors.computeIfAbsent(correspondingIntent, k -> new ArrayList<>()).add(docVectors.get(docIndex));
            docIndex++;
        }
        for (Intent intent : intents) {
            List<double[]> vs = intentToDocVectors.get(intent);
            if (vs == null || vs.isEmpty()) {
                intent.vector = new double[vocabIndex.size()]; // zero vector
            } else {
                double[] avg = new double[vocabIndex.size()];
                for (double[] v : vs) {
                    for (int i = 0; i < avg.length; i++) avg[i] += v[i];
                }
                for (int i = 0; i < avg.length; i++) avg[i] /= (double) vs.size();
                normalizeInPlace(avg);
                intent.vector = avg;
            }
        }
    }

    public String reply(String userInput) {
        if (userInput == null || userInput.trim().isEmpty()) return "Please type something.";

        double[] userVec = toTfIdfVector(tokenize(userInput));
        if (userVec == null) return getFallback();

        Intent best = null;
        double bestScore = 0.0;
        for (Intent intent : intents) {
            if (intent.vector == null) continue;
            double sim = cosineSimilarity(userVec, intent.vector);
            if (sim > bestScore) {
                bestScore = sim;
                best = intent;
            }
        }

        if (best == null || bestScore < THRESHOLD) {
            return getFallback();
        }
        if (best.responses.isEmpty()) return getFallback();
        String resp = best.responses.get(rnd.nextInt(best.responses.size()));
        return resp + String.format(" (matched: %s, score=%.3f)", best.name != null ? best.name : "?", bestScore);
    }

    private String getFallback() {
        for (Intent it : intents) {
            if (it.name != null && it.name.equalsIgnoreCase("Fallback") && !it.responses.isEmpty()) {
                return it.responses.get(rnd.nextInt(it.responses.size()));
            }
        }
        return "Sorry, I didn't understand that. Could you rephrase?";
    }


    private List<String> tokenize(String text) {
        if (text == null) return Collections.emptyList();
        String clean = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ");
        String[] parts = clean.split("\\s+");
        List<String> toks = new ArrayList<>();
        for (String p : parts) {
            String t = p.trim();
            if (t.isEmpty()) continue;
            if (STOPWORDS.contains(t)) continue;
            toks.add(t);
        }
        return toks;
    }


    private double[] toTfIdfVector(List<String> tokens) {
        if (tokens == null || tokens.isEmpty() || vocabIndex.isEmpty()) return null;
        double[] vec = new double[vocabIndex.size()];
        Map<String, Integer> counts = new HashMap<>();
        int max = 0;
        for (String t : tokens) {
            if (!vocabIndex.containsKey(t)) continue;
            int c = counts.getOrDefault(t, 0) + 1;
            counts.put(t, c);
            if (c > max) max = c;
        }
        if (max == 0) return null; // no known
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            String term = e.getKey();
            int c = e.getValue();
            int pos = vocabIndex.get(term);
            double tf = (double) c / (double) max;
            vec[pos] = tf * idf[pos];
        }
        normalizeInPlace(vec);
        return vec;
    }

    private double cosineSimilarity(double[] a, double[] b) {
        if (a == null || b == null || a.length != b.length) return 0.0;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private void normalizeInPlace(double[] v) {
        double n = 0;
        for (double x : v) n += x * x;
        if (n == 0) return;
        n = Math.sqrt(n);
        for (int i = 0; i < v.length; i++) v[i] /= n;
    }
}
