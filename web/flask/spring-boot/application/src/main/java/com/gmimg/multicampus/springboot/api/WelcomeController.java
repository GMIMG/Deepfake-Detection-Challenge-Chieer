package com.gmimg.multicampus.springboot.api;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;


@Controller
public class WelcomeController {

    @RequestMapping("/welcome")
    public String welcome() { 
        return "welcome";
    }
    @GetMapping("/welcome2")
    public String welcome2() { 
        return "welcome2";
    }
}