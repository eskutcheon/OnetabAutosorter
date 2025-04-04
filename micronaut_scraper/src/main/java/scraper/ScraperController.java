package scraper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.micronaut.http.annotation.Body;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Post;

@Controller("/scrape")
public class ScraperController {
    private final ScraperService service;

    public ScraperController(ScraperService service) {
        this.service = service;
    }

    @Post
    public Map<String, Object> scrapeSingle(@Body Map<String, String> body) {
        String url = body.get("url");
        if (url == null || url.isBlank()) return Map.of("error", "URL is required");
        ScrapeResult result = service.fetch(url);
        return Map.of(
            "url", result.url,
            "title", result.title,
            "description", result.description,
            "summary", result.summary.strip()
        );
    }

    // @Post("/batch")
    // public List<Map<String, Object>> scrapeBatch(@Body Map<String, List<String>> body) {
    //     List<String> urls = body.getOrDefault("urls", List.of());
    //     return urls.stream().map(url -> {
    //         ScrapeResult res = service.fetch(url);
    //         /*? NOTE: Java can’t infer the generic type T because Map.of(...) creates a Map<String, Object>, but the List<Map<String, String>>
    //             that scrapeBatch is expected to return is more specific.
    //         */
    //         return Map.of(
    //             "url", res.url,
    //             "title", res.title,
    //             "description", res.description,
    //             "summary", res.summary.strip()
    //         );
    //     }).collect(Collectors.toList());
    // }
    @Post("/batch")
    public List<Map<String, String>> scrapeBatch(@Body Map<String, List<String>> body) {
        List<String> urls = body.getOrDefault("urls", List.of());
        return urls.stream().map(
            url -> {
                ScrapeResult res = service.fetch(url);
                Map<String, String> result = new HashMap<>();
                result.put("url", res.url);
                result.put("title", res.title);
                result.put("description", res.description);
                result.put("summary", res.summary.strip());
                return result;
            }
        ).collect(Collectors.toList());
    }
}
