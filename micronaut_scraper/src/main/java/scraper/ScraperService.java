package scraper;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import jakarta.inject.Singleton;
// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;


@Singleton
public class ScraperService {
    // reference if this doesn't work: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Java
        //? NOTE: this should be more memory-efficient
    public static int levenshtein(String s0, String s1) {
        int len0 = s0.length() + 1;
        int len1 = s1.length() + 1;
        int[] cost = new int[len0];
        int[] newcost = new int[len0];
        for (int i = 0; i < len0; i++)
            cost[i] = i;
        for (int j = 1; j < len1; j++) {
            newcost[0] = j;
            for (int i = 1; i < len0; i++) {
                int match = (s0.charAt(i - 1) == s1.charAt(j - 1)) ? 0 : 1;
                int cost_replace = cost[i - 1] + match;
                int cost_insert = cost[i] + 1;
                int cost_delete = newcost[i - 1] + 1;
                newcost[i] = Math.min(Math.min(cost_insert, cost_delete), cost_replace);
            }
            int[] swap = cost;
            cost = newcost;
            newcost = swap;
        }
        return cost[len0 - 1];
    }

    public static boolean isSimilarText(String a, String b, int threshold) {
        if (a == null || b == null || a.isEmpty() || b.isEmpty()) return false;
        // limit max length for comparison to 50 characters for performance and relevance
        int maxLen = Math.min(50, Math.min(a.length(), b.length()));
        a = a.substring(0, maxLen).trim();
        b = b.substring(0, maxLen).trim();
        int dist = levenshtein(a, b);
        int maxPossible = Math.max(a.length(), b.length());
        int ratio = 100 - (int) ((double) dist / maxPossible * 100);
        return ratio >= threshold;
    }


    private final Map<String, ScrapeResult> cache = new ConcurrentHashMap<>();

    // private static final Logger logger = LoggerFactory.getLogger(ScraperService.class);
    public ScrapeResult fetch(String url) {
        // Check cache first to avoid redundant scraping
        if (cache.containsKey(url)) {
            return cache.get(url);
        }
        // If not in cache, perform scraping
        ScrapeResult result = new ScrapeResult(url);
        try {
            Document doc = Jsoup.connect(url)
                .userAgent("Mozilla/5.0")
                .timeout(5000) // increasing this to 5 seconds to reduce early exits
                .get();
            // note the last added string for checking for near-duplicate text
            String lastAdded = "";
            // scrape the title tag and save to results
            Element titleTag = doc.selectFirst("title");
            if (titleTag != null) {
                //result.title = titleTag.text();
                //logger.info("Scraped title: {}", result.title);
                //System.out.println("Scraped title: " + result.title); // Debug log for title
                String title = titleTag.text();
                result.title = title;
                if (!isSimilarText(lastAdded, title, 70)) {
                    result.appendToSummary(title);
                    lastAdded = title;
                }
            }
            // check meta tags for description and other relevant information
            for (String metaKey : new String[]{"description", "og:description", "twitter:description"}) {
                Element meta = doc.selectFirst("meta[name=" + metaKey + "], meta[property=" + metaKey + "]");
                if (meta != null) {
                    //result.description = meta.attr("content");
                    //logger.info("Scraped {}: {}", metaKey, result.description);
                    //System.out.println("Scraped " + metaKey + ": " + result.description); // Debug log for description
                    //result.appendToSummary(meta.attr("content"));
                    String content = meta.attr("content");
                    if (!isSimilarText(lastAdded, content, 70)) {
                        result.description = content;
                        result.appendToSummary(content);
                        lastAdded = content;
                    }
                }
            }
            //result.appendToSummary(result.title);
            // scrape the first and last paragraph text for summary
            var paragraphs = doc.select("p");
            if (!paragraphs.isEmpty()) {
                //result.appendToSummary(paragraphs.first().text());
                //result.appendToSummary(paragraphs.last().text());
                // logger.info("Scraped summary paragraphs: {} ... {}", paragraphs.first().text(), paragraphs.last().text());
                //System.out.println("Scraped summary paragraphs: " + paragraphs.first().text() + " ... " + paragraphs.last().text()); // Debug log for summary
                String first = paragraphs.first().text();
                String last = paragraphs.last().text();
                if (!isSimilarText(lastAdded, first, 70)) {
                    result.appendToSummary(first);
                    lastAdded = first;
                }
                if (!isSimilarText(lastAdded, last, 70))
                    result.appendToSummary(last);
            }
        }
        catch (Exception e) {
            result.appendToSummary("Failed to scrape: " + e.getMessage());
        }

        cache.put(url, result);
        return result;
    }
}
