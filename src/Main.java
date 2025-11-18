import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) {
        String intentsPath = "data/intents.txt";
        if (args.length > 0) intentsPath = args[0];

        ChatbotTFIDF bot;
        try {
            bot = new ChatbotTFIDF(intentsPath);
        } catch (IOException e) {
            System.err.println("Failed to load intents: " + e.getMessage());
            return;
        }

        System.out.println("TF-IDF Chatbot ready! Type 'exit' to quit.");
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        while (true) {
            try {
                System.out.print("> ");
                String line = br.readLine();
                if (line == null) break;
                line = line.trim();
                if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
                    System.out.println("Goodbye!");
                    break;
                }
                if (line.isEmpty()) {
                    System.out.println("Bot: Please type something.");
                    continue;
                }
                String reply = bot.reply(line);
                System.out.println("Bot: " + reply);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
