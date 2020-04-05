package com.gmimg.multicampus.springboot.service;

import javax.management.RuntimeErrorException;

import com.gmimg.multicampus.springboot.domain.Book;
import com.gmimg.multicampus.springboot.domain.BookRepository;

import org.aspectj.lang.annotation.Before;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.SpringBootTest.WebEnvironment;

@SpringBootTest(webEnvironment = WebEnvironment.NONE)
public class BookServiceTest {
    @Autowired
    BookService bookService;

    // @org.junit.Before
    // public void setUp() {
    //     BookRepository bookRepository = new StubBookRepository();
    //     bookService = new BookServiceInpl(bookRepository);
    // }
    // @ExtendWith()
    // @Test(expected=RuntimeException.class)
    public void testFindById() {
        Long id = 1L;
        bookService.findById(id)
            .orElseThrow(() -> new RuntimeException("Not found"));
    }
}